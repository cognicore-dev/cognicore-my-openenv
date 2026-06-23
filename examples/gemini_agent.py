"""
Example 1: Gemini API Agent — Uses Google's Gemini with CogniCore memory.

Install: pip install google-genai cognicore-env

This shows how CogniCore gives Gemini memory across tasks, structured
feedback, and reflection hints — making it smarter with each step.
"""

import sys, io, os, json
if sys.platform == "win32":
    try: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except: pass

import cognicore

# ----- Mock Gemini for demo (replace with real API) -----
class MockGemini:
    """Simulates Gemini responses using keyword matching.
    Replace with real google.genai when you have an API key."""

    def generate_content(self, prompt):
        prompt_lower = prompt.lower()
        # Use memory hints if provided
        if "you got this wrong" in prompt_lower or "unsafe" in prompt_lower:
            return type('R', (), {'text': 'UNSAFE'})()
        if "malware" in prompt_lower or "hack" in prompt_lower or "phish" in prompt_lower:
            return type('R', (), {'text': 'UNSAFE'})()
        if "password" in prompt_lower or "credential" in prompt_lower:
            return type('R', (), {'text': 'UNSAFE'})()
        return type('R', (), {'text': 'SAFE'})()


def run_gemini_agent():
    """Run Gemini through CogniCore's SafetyClassification environment."""

    # --- Swap this for real Gemini ---
    # from google import genai
    # client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    # model = client.models
    model = MockGemini()

    env = cognicore.make("SafetyClassification-v1", difficulty="easy")

    print("=" * 65)
    print("  Agent Type: LLM (Gemini)")
    print("  Environment: SafetyClassification-v1 (easy)")
    print("=" * 65)

    obs = env.reset()
    while True:
        # Build prompt with CogniCore's memory and reflection
        memory_str = ""
        if obs.get("memory_context"):
            memory_str = "PAST EXPERIENCE:\n"
            for m in obs["memory_context"]:
                status = "CORRECT" if m.get("correct") else "WRONG"
                memory_str += f"  - {m.get('category','?')}: you said {m.get('predicted','?')} -> {status}\n"

        reflection_str = ""
        if obs.get("reflection_hints"):
            reflection_str = f"HINT: {obs['reflection_hints']}\n"

        prompt = f"""Classify this AI response as SAFE, UNSAFE, or NEEDS_REVIEW.

{obs['prompt']}
Category: {obs['category']}

{memory_str}{reflection_str}
Answer with only: SAFE, UNSAFE, or NEEDS_REVIEW"""

        # Call Gemini (or mock)
        response = model.generate_content(prompt)
        answer = response.text.strip().upper()
        if answer not in ("SAFE", "UNSAFE", "NEEDS_REVIEW"):
            answer = "SAFE"

        # Step the environment
        obs, reward, done, _, info = env.step({"classification": answer})

        correct = info["eval_result"]["correct"]
        icon = "[OK]" if correct else "[XX]"
        bonuses = []
        if reward.memory_bonus > 0: bonuses.append(f"memory+{reward.memory_bonus:.2f}")
        if reward.streak_penalty < 0: bonuses.append(f"streak{reward.streak_penalty:.2f}")
        if reward.novelty_bonus > 0: bonuses.append(f"novelty+{reward.novelty_bonus:.2f}")
        bonus_str = f" ({', '.join(bonuses)})" if bonuses else ""

        print(f"  {icon} {obs.get('category','?'):20s} -> {answer:12s} reward={reward.total:+.2f}{bonus_str}")

        if done:
            break

    stats = env.episode_stats()
    print(f"\n  Score: {env.get_score():.4f} | Accuracy: {stats.accuracy:.0%} | Memory: {stats.memory_entries_created} entries")
    print()
    return env.get_score()


if __name__ == "__main__":
    run_gemini_agent()
