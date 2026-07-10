import openai

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="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"
)

response = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=[{"role": "user", "content": "Say hello!"}],
    max_tokens=10
)

print(response.choices[0].message.content)
