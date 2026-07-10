from cognicore_benchmarks.common.llm_client import LLMClient

client = LLMClient(model_name="gemini-2.5-flash")
res = client.generate("Is the sky blue?", system_prompt=None)
print("gemini-2.5-flash:", res)

client2 = LLMClient(model_name="gemini-1.5-flash")
res2 = client2.generate("Is the sky blue?", system_prompt=None)
print("gemini-1.5-flash:", res2)
