import requests
import os

# To use an environment variable, you would set it like this:
# os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-..."
# Then you can access it in your code like this:
API_KEY = os.environ.get("OPENROUTER_API_KEY", "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")

# --- IMPORTANT: Replace "YOUR_API_KEY_HERE" with your actual key ---
# You can get your key from: https://openrouter.ai/keys

url = "https://openrouter.ai/api/v1/embeddings"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}
data = {
    "model": "openai/text-embedding-3-small",
    "input": "Test text to embed",
}

try:
    resp = requests.post(url, headers=headers, json=data)
    resp.raise_for_status()  # This will raise an HTTPError for bad responses (4xx or 5xx)

    print("Status Code:", resp.status_code)
    print("Raw Response:", resp.text)

    result = resp.json()
    embedding = result["data"][0]["embedding"]
    print("✅ Got embedding:", embedding[:10])
    print("Dimension:", len(embedding))

except requests.exceptions.HTTPError as err:
    print(f"❌ API call failed with HTTP error: {err}")
    print("Response body:", resp.text)
except requests.exceptions.RequestException as e:
    print(f"❌ An error occurred: {e}")
except KeyError as e:
    print(f"❌ Failed to parse response, missing key: {e}")
