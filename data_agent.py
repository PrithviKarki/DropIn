import asyncio
import os
import json
import sys
from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END

# Import the Gemini SDK
from google import genai
from google.genai import types, errors

# Import Tenacity for rate-limiting
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception

# Import MCP Client libraries
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Define the State for our LangGraph
class AgentState(TypedDict):
    messages: Annotated[Sequence[dict], operator.add]
    query: str


def _is_retryable_api_error(exc: BaseException) -> bool:
    return isinstance(exc, errors.APIError) and getattr(exc, "code", None) in (429, 500, 503)


class DataRetrievalAgent:
    def __init__(self, model: str = "gemini-2.5-flash"):
        # Matched Experimental Control: Default to gemini-2.5-flash
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = model

        # Build the LangGraph
        workflow = StateGraph(AgentState)

        # Define the nodes
        workflow.add_node("llm", self.call_llm)
        workflow.add_node("tools", self.execute_mcp_tool)

        # Define the edges (Flow)
        workflow.set_entry_point("llm")
        workflow.add_conditional_edges("llm", self.should_continue)
        workflow.add_edge("tools", "llm")  # After tool runs, go back to LLM

        self.app = workflow.compile()
        self.mcp_session = None

        # Telemetry & Token Usage Tracking
        self.last_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "llm_calls": 0,
            "mcp_tool_calls": [],
            "mcp_tool_success": True,
        }

    @retry(
        retry=retry_if_exception(_is_retryable_api_error),
        wait=wait_exponential(multiplier=2, min=15, max=90),
        stop=stop_after_attempt(6),
        before_sleep=lambda retry_state: print(
            f"⚠️  [API Rate Limit / Error] Attempt {retry_state.attempt_number} failed: {retry_state.outcome.exception()}. "
            f"Waiting {round(retry_state.next_action.sleep, 1)}s before retry..."
        ),
    )
    async def call_llm(self, state: AgentState):
        """Node 1: The LLM thinks and decides if it needs to call a tool."""
        print("🤖 Agent is thinking...")

        # Convert MCP tools to Gemini tools format
        mcp_tools = await self.mcp_session.list_tools()
        gemini_tools = []
        for t in mcp_tools.tools:
            gemini_tools.append(
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=t.name,
                            description=t.description,
                            parameters=t.input_schema if getattr(t, "input_schema", None) else None
                        )
                    ]
                )
            )

        # Create system instructions to strictly guide the agent
        config = types.GenerateContentConfig(
            tools=gemini_tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            system_instruction="""You are the Data Retrieval Agent. 
            Your only job is to write SQL to answer the user's question. 
            Step 1: Always call get_database_schema to understand the tables. 
            Step 2: Call query_database with your SQL. 
            Step 3: Return the factual result to the user. Do not analyze it."""
        )

        # Send to Gemini
        response = self.client.models.generate_content(
            model=self.model,
            contents=state["messages"],
            config=config
        )

        # Track token usage across all internal tool-calling turns
        usage = getattr(response, "usage_metadata", None)
        if usage:
            self.last_usage["prompt_tokens"] += getattr(usage, "prompt_token_count", 0) or 0
            self.last_usage["completion_tokens"] += getattr(usage, "candidates_token_count", 0) or 0
            self.last_usage["total_tokens"] += getattr(usage, "total_token_count", 0) or 0
        self.last_usage["llm_calls"] += 1

        # Append the actual Content object from the model, not the full response wrapper
        return {"messages": [response.candidates[0].content]}

    def should_continue(self, state: AgentState):
        """Condition: Checks if the LLM decided to call a tool or if it is finished."""
        last_message = state["messages"][-1]

        # Check if the LLM returned a function call in its parts
        if last_message.parts and getattr(last_message.parts[0], "function_call", None):
            print(f"🔄 Agent decided to use a tool: {last_message.parts[0].function_call.name}")
            return "tools"

        # Otherwise, we are done
        return END

    async def execute_mcp_tool(self, state: AgentState):
        """Node 2: Executes the requested tool on the MCP server."""
        last_message = state["messages"][-1]
        tool_call = last_message.parts[0].function_call

        print(f"🛠️ Executing MCP Tool: {tool_call.name}...")

        args = tool_call.args if hasattr(tool_call, "args") else {}
        result = await self.mcp_session.call_tool(tool_call.name, arguments=args)
        result_text = result.content[0].text if result.content else ""

        is_error = bool(getattr(result, "isError", False)) or result_text.startswith("Error") or result_text.startswith("Security Error")
        if is_error:
            self.last_usage["mcp_tool_success"] = False

        self.last_usage["mcp_tool_calls"].append({
            "tool_name": tool_call.name,
            "arguments": dict(args) if args else {},
            "success": not is_error,
        })

        # Format the result back for the LLM
        tool_response = types.Content(
            role="user",
            parts=[types.Part.from_function_response(
                name=tool_call.name,
                response={"result": result_text}
            )]
        )

        return {"messages": [tool_response]}

    async def run(self, query: str):
        """Main loop that connects to the MCP Server and runs the LangGraph."""
        # Reset telemetry for this run
        self.last_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "llm_calls": 0,
            "mcp_tool_calls": [],
            "mcp_tool_success": True,
        }

        server_params = StdioServerParameters(
            command=sys.executable,
            args=["mcp_server.py"]
        )

        print("🔗 Connecting to MCP Server...")
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self.mcp_session = session
                print("✅ Connected to Database Server.")

                # Start the LangGraph execution
                initial_state = {
                    "messages": [types.Content(role="user", parts=[types.Part.from_text(text=query)])],
                    "query": query
                }

                final_state = await self.app.ainvoke(initial_state)

                print("\n" + "=" * 50)
                print("FINAL ANSWER FROM AGENT:")
                print("=" * 50)
                final_content = final_state["messages"][-1]
                answer = final_content.parts[0].text if final_content.parts else "No text returned."
                print(answer)
                print("=" * 50)
                return answer


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    agent = DataRetrievalAgent()

    test_question = "What is the GPA and attendance for the student with ID 4?"
    if len(sys.argv) > 1:
        test_question = " ".join(sys.argv[1:])

    print(f"User Query: {test_question}\n")
    asyncio.run(agent.run(test_question))
    print(f"\n📊 Token & Tool Usage Telemetry: {json.dumps(agent.last_usage, indent=2)}")
