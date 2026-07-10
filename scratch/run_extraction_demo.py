import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cognicore.memory.extractor import extract_memories

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MOCK_TRANSCRIPT = """
User: Hey, I want to build a new dashboard for my app.
Agent: Sure! What tech stack are we using?
User: We're using Next.js on the frontend, but I absolutely do not want to use Tailwind CSS. I find it too messy. Let's stick to standard CSS modules.
Agent: Got it. Standard CSS modules it is. What about the backend?
User: The backend is a standard Node.js Express server. Also, make sure all buttons have rounded corners, it's a strict design rule for my brand.
Agent: Understood. I will ensure buttons have rounded corners and we use standard CSS modules.
"""

def main():
    print("==========================================")
    print("CogniCore Automated Memory Extraction Demo")
    print("==========================================")
    print("\nReading transcript...")
    print(MOCK_TRANSCRIPT)
    
    print("\n[Running Extractor... (Calling LLM)]")
    extracted = extract_memories(MOCK_TRANSCRIPT, agent_id="demo_agent")
    
    print("\n==========================================")
    print("Extraction Results")
    print("==========================================")
    
    if not extracted:
        print("No memories extracted or extraction failed.")
        return
        
    for idx, mem in enumerate(extracted):
        print(f"{idx+1}. [{mem.get('memory_type', 'semantic').upper()}] {mem.get('text')}")
        
    print("\nMemories have been embedded and stored in the ChromaDB vector engine!")
    
if __name__ == "__main__":
    main()
