import os
import google.generativeai as genai
from dotenv import load_dotenv

# Carrega a chave
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print("--- Modelos disponíveis para a sua chave ---")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Erro ao consultar a API: {e}")