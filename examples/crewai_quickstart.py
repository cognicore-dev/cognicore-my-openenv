import os
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
from cognicore.integrations.crewai import cognicore_crewai_tools

def main():
    # 1. Load the CogniCore Tools
    # This array includes Remember, Recall, Reflect, and ThreatScan tools
    # All tools share a unified memory context!
    memory_tools = cognicore_crewai_tools()

    # 2. Assign the tools to your CrewAI Agent
    researcher = Agent(
        role='Senior Data Analyst',
        goal='Analyze incoming text and rely on past experiences to avoid mistakes.',
        backstory="An expert analyst who uses persistent memory to recall past failures and successes.",
        verbose=True,
        allow_delegation=False,
        tools=memory_tools,
        llm="openrouter/openai/gpt-4o-mini"
    )

    # 3. Define a task that requires memory
    task = Task(
        description="Search your memory (using cognicore_recall) for any past advice about handling API rate limits. If you find none, write a quick guide and use cognicore_remember to store it.",
        expected_output="A summary of the recalled memory or the newly stored guide.",
        agent=researcher
    )

    # 4. Run the Crew
    crew = Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential
    )

    print("Starting the Crew. Watch as it interacts with CogniCore memory!")
    result = crew.kickoff()
    print(result)

if __name__ == "__main__":
    main()
