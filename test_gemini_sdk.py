import google.generativeai as genai

try:
    genai.configure(api_key="AQ.Ab8RN6Lpx56k6BtC1fyL3Pov1-94MIJ1_9tIgQkoEzDoQT5RBg")
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Is the sky blue?")
    print("Success:", response.text)
except Exception as e:
    print("Error:", e)
