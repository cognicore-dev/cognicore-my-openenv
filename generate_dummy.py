import os
import json
import time

def generate_dummy_data():
    base_dir = "./cognicore_data"
    os.makedirs(base_dir, exist_ok=True)
    
    # Agent 1: Good memory
    agent1_dir = os.path.join(base_dir, "agent-123")
    os.makedirs(agent1_dir, exist_ok=True)
    
    agent1_memories = [
        {
            "entry_id": "mem_001",
            "text": "When user asks for python code, prefer using type hints.",
            "category": "preference",
            "scope": "user",
            "scope_id": "global",
            "state": "active",
            "timestamp": time.time(),
            "utility_score": 0.85,
            "used_count": 10,
            "positive_outcomes": 9,
            "negative_outcomes": 1,
            "ignored_count": 0,
            "retrieval_count": 12
        },
        {
            "entry_id": "mem_002",
            "text": "Always validate the input string before parsing JSON.",
            "category": "procedural",
            "scope": "global",
            "scope_id": "global",
            "state": "verified",
            "timestamp": time.time(),
            "utility_score": 0.95,
            "used_count": 25,
            "positive_outcomes": 25,
            "negative_outcomes": 0,
            "ignored_count": 2,
            "retrieval_count": 30
        }
    ]
    
    with open(os.path.join(agent1_dir, "memory.json"), "w") as f:
        json.dump(agent1_memories, f, indent=2)

    # Agent 2: Has a negative transfer memory
    agent2_dir = os.path.join(base_dir, "agent-456")
    os.makedirs(agent2_dir, exist_ok=True)
    
    agent2_memories = [
        {
            "entry_id": "mem_003",
            "text": "The database uses port 5432.",
            "category": "semantic",
            "scope": "global",
            "scope_id": "global",
            "state": "active",
            "timestamp": time.time() - 86400,
            "utility_score": -0.4,
            "used_count": 8,
            "positive_outcomes": 2,
            "negative_outcomes": 6,  # High negative transfer!
            "ignored_count": 1,
            "retrieval_count": 10
        }
    ]
    
    with open(os.path.join(agent2_dir, "memory.json"), "w") as f:
        json.dump(agent2_memories, f, indent=2)
        
    print("Dummy data generated at ./cognicore_data/")

if __name__ == "__main__":
    generate_dummy_data()
