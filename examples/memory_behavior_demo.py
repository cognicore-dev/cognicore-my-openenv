from cognicore.integrations.langchain import cognicore_tools, _get_semantic_memory
from cognicore.integrations.langchain import cognicore_tools
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
import os

def main():
    print("=== CogniCore: Memory Changes Behavior Demo ===")
    
    llm = ChatOpenAI(
        temperature=0, 
        model="openai/gpt-4o-mini",
        api_key=os.environ.get("OPENROUTER_API_KEY", "mock_key"),
        base_url="https://openrouter.ai/api/v1"
    )
    tools = cognicore_tools()
    
    agent = create_react_agent(llm, tools=tools)
    
    trick_question = "What is the URL for the internal HR company portal? (Search your memory using cognicore_recall if you don't know it)."
    
    print("\n[Phase 1] Asking the agent without any prior memory...")
    try:
        response1 = agent.invoke({"messages": [("user", trick_question)]})
        print("Agent says:", response1["messages"][-1].content)
    except Exception as e:
        print("Agent failed or hallucinated.")

    print("\n[Phase 2] Injecting ground truth into CogniCore's Semantic Memory...")
    memory_db = _get_semantic_memory()
    memory_db.store({
        "text": "The internal HR portal was moved last week. The new URL is https://internal.acme.corp/hr-v2",
        "category": "company_knowledge",
        "correct": True,
        "predicted": "knowledge_update"
    })
    print("Memory injected.")

    print("\n[Phase 3] Asking the exact same question again...")
    try:
        response2 = agent.invoke({"messages": [("user", trick_question)]})
        print("Agent says:", response2["messages"][-1].content)
        print("\nNotice how the agent's behavior completely changed without altering its prompt or model!")
    except Exception as e:
        print("Error running agent.")

if __name__ == "__main__":
    main()
