import sqlite3
from mcp.server.mcpserver import MCPServer

# Initialize the MCPServer
mcp = MCPServer("DropIn-Database-Server")
DB_NAME = "dropin_synthetic.db"

@mcp.tool()
def get_database_schema() -> str:
    """
    Returns the schema of the DropIn SQLite database. 
    Use this tool to understand the table structure before writing SQL queries.
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Query to get the schema of all tables
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        
        if not tables:
            return "No tables found in the database."
            
        schema = "\n\n".join([table[0] for table in tables if table[0]])
        return f"Database Schema:\n{schema}"
    except Exception as e:
        return f"Error reading schema: {e}"

@mcp.tool()
def query_database(sql_query: str) -> str:
    """
    Executes a READ-ONLY SQL query (SELECT) on the database and returns the results.
    Do not use this for INSERT, UPDATE, or DELETE operations.
    """
    # Basic security guardrail to prevent accidental data deletion by the agent
    if not sql_query.strip().upper().startswith("SELECT"):
        return "Error: Security violation. Only SELECT queries are allowed."
        
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row # Allows us to access columns by name
        cursor = conn.cursor()
        
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return "Query executed successfully but returned 0 results."
            
        # Format results as a readable CSV-like string for the LLM
        columns = rows[0].keys()
        result_lines = [" | ".join(columns)]
        result_lines.append("-" * 50)
        
        for row in rows:
            result_lines.append(" | ".join(str(row[col]) for col in columns))
            
        return "\n".join(result_lines)
        
    except sqlite3.Error as e:
        return f"SQLite Database error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

if __name__ == "__main__":
    import sys
    print("Starting DropIn SQLite MCP Server...", file=sys.stderr)
    # This runs the server using standard input/output (stdio), which is 
    # the standard communication protocol for MCP.
    mcp.run()

