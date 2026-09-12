import os
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Load environment variables from .env
load_dotenv()

# Initialize the client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Decorator to retry on ResourceExhausted (Rate Limit 429) errors
@retry(
    retry=retry_if_exception_type(errors.APIError),
    wait=wait_exponential(multiplier=2, min=2, max=30), # Waits 2s, 4s, 8s, up to 30s
    stop=stop_after_attempt(5)
)
def generate_response_with_backoff(prompt: str, model: str = "gemini-2.5-flash"):
    """Calls the Gemini API and automatically retries if rate limited."""
    print(f"Sending prompt to {model}...")
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    return response.text

if __name__ == "__main__":
    # Test the connection
    try:
        result = generate_response_with_backoff("Say 'Infrastructure setup complete!'")
        print(f"API Response: {result}")
    except Exception as e:
        print(f"Failed to connect: {e}") 