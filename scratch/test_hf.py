from sentence_transformers import CrossEncoder
try:
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("Successfully loaded cross-encoder!")
except Exception as e:
    print(f"Failed: {e}")
