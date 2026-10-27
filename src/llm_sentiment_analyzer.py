import json
import os
import time
import urllib.request
import ssl
from pathlib import Path
from typing import Literal
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
# Load environment variables from .env in project root
load_dotenv(BASE_DIR / ".env")

DEFAULT_INPUT_FILE = str(BASE_DIR / "data" / "processed" / "preprocessed_comments.json")
DEFAULT_OUTPUT_DIR = str(BASE_DIR / "outputs")

class SentimentOutput(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        ...,
        description="Strict sentiment classification: must be one of 'positive', 'negative', 'neutral'"
    )
    score: Literal[1, 0, -1] = Field(
        ...,
        description="Sentiment numerical score: 1 for positive, -1 for negative, 0 for neutral"
    )

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are an expert citizen feedback sentiment analyzer for e-governance public policies and grievances.
Analyze the citizen comment and classify its sentiment into exactly one of three categories:
- 'positive' (score: 1) -> Expresses support, satisfaction, praise, or constructive appreciation.
- 'negative' (score: -1) -> Expresses dissatisfaction, complaints, delay, portal glitches, or criticism.
- 'neutral' (score: 0) -> Neutral query, factual statement, general policy clause, or administrative notice.

You MUST respond in valid JSON matching this exact schema:
{
  "sentiment": "positive" | "negative" | "neutral",
  "score": 1 | 0 | -1
}
Do not include any explanation, markdown formatting, or text outside the JSON object.
"""
def analyze_comment_with_llm(comment_text, api_key=None, model=DEFAULT_MODEL, max_retries=3):
    """
    Calls the Groq API and strictly validates the response using Pydantic.
    Returns: (sentiment: str, score: int)
    """
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key or api_key == "YOUR_GROQ_API_KEY_HERE":
        raise ValueError("GROQ_API_KEY is not set. Please specify it in the .env file or environment variables.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Citizen Comment: \"{comment_text}\""}
        ],
        "temperature": 0.0,  # Deterministic output
        "response_format": {"type": "json_object"}  # Enforce JSON mode
    }

    req_data = json.dumps(payload).encode("utf-8")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(GROQ_API_URL, data=req_data, headers=headers)
            with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            raw_content = result["choices"][0]["message"]["content"]

            # Strict Pydantic Validation
            parsed: SentimentOutput = SentimentOutput.model_validate_json(raw_content)

            # Ensure score strictly aligns with sentiment
            expected_score = 1 if parsed.sentiment == "positive" else (-1 if parsed.sentiment == "negative" else 0)
            final_score = parsed.score if parsed.score == expected_score else expected_score

            return parsed.sentiment, final_score

        except urllib.error.HTTPError as e:
            if e.code == 429:  # Rate limited
                wait_time = (attempt + 1) * 3
                print(f"    [!] Rate limit reached (429). Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            elif attempt == max_retries - 1:
                print(f"    [-] HTTP Error {e.code}: {e.read().decode('utf-8', errors='ignore')}")
                return "neutral", 0
        except (ValidationError, json.JSONDecodeError) as e:
            if attempt == max_retries - 1:
                print(f"    [-] Pydantic validation error: {e}")
                return "neutral", 0
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"    [-] Error: {e}")
                return "neutral", 0
    return "neutral", 0

def run_sentiment_pipeline(
    input_file=None,
    output_dir=None,
    api_key=None,
    model=DEFAULT_MODEL,
    limit=None
):
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if input_file is None:
        input_file = DEFAULT_INPUT_FILE
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("[*] STARTING LLM SENTIMENT ANALYSIS VIA GROQ & PYDANTIC")
    print(f"[*] Input Dataset : {input_file}")
    print(f"[*] Target Model  : {model}")
    print("=" * 65 + "\n")

    if not os.path.exists(input_file):
        print(f"[-] Input file not found: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    if limit:
        records = records[:limit]
        print(f"[*] Testing on first {limit} records as requested.")
    else:
        print(f"[*] Processing all {len(records)} records...")

    total = len(records)
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}

    for idx, item in enumerate(records, start=1):
        text_to_analyze = item.get("cleaned_comment") or item.get("raw_comment") or ""

        print(f"[{idx}/{total}] ID {item['id']} ({item.get('source_portal', 'Portal')}): Analyzing...", end="", flush=True)

        sentiment, score = analyze_comment_with_llm(text_to_analyze, api_key=api_key, model=model)

        item["sentiment"] = sentiment
        item["sentiment_score"] = score

        sentiment_counts[sentiment] += 1
        print(f" -> [{sentiment.upper()} | score: {score}]")
        time.sleep(0.4)

    json_out = os.path.join(output_dir, "sentiment_analyzed_comments.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    df = pd.DataFrame(records)
    csv_out = os.path.join(output_dir, "sentiment_analyzed_comments.csv")
    df.to_csv(csv_out, index=False, encoding="utf-8")

    print("\n" + "=" * 65)
    print("[OK] SENTIMENT ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"[OK] Total Records Processed : {total}")
    print(f"[OK] Positive Feedback (1)  : {sentiment_counts['positive']}")
    print(f"[OK] Negative Feedback (-1) : {sentiment_counts['negative']}")
    print(f"[OK] Neutral / Factual (0)   : {sentiment_counts['neutral']}")
    print(f"[OK] JSON Output             : {json_out}")
    print(f"[OK] CSV Output (Power BI)   : {csv_out}")
    print("=" * 65)

if __name__ == "__main__":
    run_sentiment_pipeline(
        limit=None 
    )
