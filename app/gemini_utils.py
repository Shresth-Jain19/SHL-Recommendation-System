from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === FETCH API KEY ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# === CONFIGURE GEMINI ===
client = genai.Client(api_key=GEMINI_API_KEY)

# === PROMPT AND MODEL ===
model_id = "gemini-2.5-flash-preview-04-17"

def get_query_from_url(url: str) -> str:
    prompt = f"""
Visit this job URL and read the full job description carefully:
{url}

Now generate a **natural language search query** as if someone is trying to find the best SHL assessment(s) for this role.

Include:
- Job title
- Key skills (technical + cognitive if any)
- Level of the role
- Time constraint if mentioned
- Output must be ≤ 2 sentences, plain text only
"""

    try:
        response = client.models.generate_content(
            contents=prompt,
            model=model_id,
            config=GenerateContentConfig(
                tools=[Tool(google_search=GoogleSearch())], response_modalities=["TEXT"]
            ),
        )
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ Gemini failed: {e}")
        return (
            "Entry-level role, basic technical and cognitive skills, under 30 minutes"
        )