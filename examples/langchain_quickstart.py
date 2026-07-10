import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*create_react_agent.*")

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from cognicore.integrations.langchain import cognicore_tools, CogniCoreCallbackHandler

def main():
    # 1. Setup the LLM
    # Use OpenAI, or OpenRouter/Gemini if configured in the environment
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini", 
        temperature=0,
        api_key=os.environ.get("OPENROUTER_API_KEY", "mock_key"),
        base_url="https://openrouter.ai/api/v1"
    )
    
    # 2. Add the CogniCore Auto-Memory Handler
    # This automatically records all LLM inputs and outputs into persistent memory!
    memory_tracker = CogniCoreCallbackHandler(category_prefix="my_app")
    llm.callbacks = [memory_tracker]
    
    # 3. Load the CogniCore Agent Tools
    # This gives the agent access to Recall, Reflect, and ThreatScan tools
    tools = cognicore_tools()
    
    # 4. Initialize the Agent
    agent = create_react_agent(llm, tools=tools)
    
    print("Agent is ready! Try asking it to recall past actions.")
    
    # Run a test query
    response = agent.invoke({"messages": [("user", "Search your memory for any past successful actions.")]})
    print(response["messages"][-1].content)

if __name__ == "__main__":
    main()
