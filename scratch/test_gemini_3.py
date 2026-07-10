from cognicore_benchmarks.common.llm_client import LLMClient
client = LLMClient(model_name="gemini-3-flash-preview")
print("Client Type:", client.client_type)
print("Model Name:", client.model_name)
res = client.generate("Hello, what model are you?")
print("Result:", res)
