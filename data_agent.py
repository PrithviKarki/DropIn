import asyncio
import os
import json
from typing import TypedDict, Annotated, Sequence, List
import operator
from langgraph.graph import StateGraph, END

# Import the Gemini SDK
from google import genai
from google.genai import types

# Import MCP Client libraries
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Define the State for our LangGraph
class AgentState(TypedDict):
    messages: Annotated[Sequence[dict], operator.add]
    query: str

class DataRetrievalAgent:
    def __init__(self):
        # We use Gemini 2.5 Flash as it is fast and supports tool calling
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = "gemini-2.5-flash"
        
        # Build the LangGraph
        workflow = StateGraph(AgentState)
        
        # Define the nodes
        workflow.add_node("llm", self.call_llm)
        workflow.add_node("tools", self.execute_mcp_tool)
        
        # Define the edges (Flow)
        workflow.set_entry_point("llm")
        workflow.add_conditional_edges("llm", self.should_continue)
        workflow.add_edge("tools", "llm") # After tool runs, go back to LLM
        
        self.app = workflow.compile()
        self.mcp_session = None

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
                            # A simple mapping, assuming SQL query is a string param
                            parameters=t.input_schema if getattr(t, "input_schema", None) else None
                        )
                    ]
                )
            )

        # Create system instructions to strictly guide the agent
        config = types.GenerateContentConfig(
            tools=gemini_tools,
            system_instruction="""You are the Data Retrieval Agent. 
            Your only job is to write SQL to answer the user's question. 
            Step 1: Always call get_database_schema to understand the tables. 
            Step 2: Call query_database with your SQL. 
            Step 3: Return the factual result to the user. Do not analyze it."""
        )

        # Send to Gemini
        # We pass the history of messages
        response = self.client.models.generate_content(
            model=self.model,
            contents=state["messages"],
            config=config
        )
        
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
        
        # Call the MCP server
        result = await self.mcp_session.call_tool(
            tool_call.name, 
            arguments=tool_call.args if hasattr(tool_call, "args") else {}
        )
        
        # Format the result back for the LLM
        tool_response = types.Content(
            role="user",
            parts=[types.Part.from_function_response(
                name=tool_call.name,
                response={"result": result.content[0].text}
            )]
        )
        
        return {"messages": [tool_response]}

    async def run(self, query: str):
        """Main loop that connects to the MCP Server and runs the LangGraph."""
        # Set up the connection to our MCP server file
        server_params = StdioServerParameters(
            command="python", # Use the python executable
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
                
                print("\n" + "="*50)
                print("FINAL ANSWER FROM AGENT:")
                print("="*50)
                # Print the final text response from the LLM
                final_content = final_state["messages"][-1]
                answer = final_content.parts[0].text if final_content.parts else "No text returned."
                print(answer)
                print("="*50)

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    
    agent = DataRetrievalAgent()
    
    # Example question
    test_question = "What is the GPA and attendance for the student with ID 4?"
    if len(sys.argv) > 1:
        test_question = " ".join(sys.argv[1:])
        
    print(f"User Query: {test_question}\n")
    asyncio.run(agent.run(test_question))

