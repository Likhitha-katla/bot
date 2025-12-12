# models.py
import os
import requests
from dotenv import load_dotenv
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("⚠️ WARNING: OPENROUTER_API_KEY is missing. Check Railway Environment Variables.")

# Chat model
LLAMA_MODEL = "meta-llama/llama-3.1-8b-instruct"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embedding(text: str):
    """
    Generates embeddings using OpenRouter embedding models.
    Ensures the input is a list (required by OpenRouter).
    """

    url = "https://openrouter.ai/api/v1/embeddings"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    # Ensure input is list — required
    payload = {
        "model": EMBED_MODEL,
        "input": [text]  
    }

    response = requests.post(url, json=payload, headers=headers, timeout=30)

    # Debug on error
    if response.status_code != 200:
        print("\nEmbedding Error:")
        print("Status:", response.status_code)
        print("Response:", response.text)
        response.raise_for_status()

    return response.json()["data"][0]["embedding"]

def call_llama(system_prompt: str, user_prompt: str):
    """
    Calls LLaMA 3.1 (8B-Instruct) using OpenRouter chat endpoint.
    """

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
    }

    response = requests.post(url, json=payload, headers=headers, timeout=30)

    if response.status_code != 200:
        print("\nLLaMA API Error:")
        print("Status:", response.status_code)
        print("Response:", response.text)
        response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]
