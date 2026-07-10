import google.generativeai as genai

try:
    genai.configure(api_key="AQ.Ab8RN6Lpx56k6BtC1fyL3Pov1-94MIJ1_9tIgQkoEzDoQT5RBg")
    models = genai.list_models()
    for m in models:
        print(m.name, m.supported_generation_methods)
except Exception as e:
    print("Error:", e)
