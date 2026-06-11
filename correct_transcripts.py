# correct_transcripts.py
# ----------------------
# Stage 2 of the pipeline.
# Reads raw_text_vosk from meeting_data.csv, sends each line to Gemini
# to fix spelling and punctuation, saves the result as the 'text' column.
#
# HOW TO RUN:
#   Mac/Linux:  export GEMINI_API_KEY="your-key-here"
#   Windows:    set GEMINI_API_KEY=your-key-here
#   Then:       python correct_transcripts.py
#
#   Get a free Gemini API key from https://aistudio.google.com
#
# Time complexity:  O(N x T) — N rows, T tokens per API call
# Space complexity: O(N) — full DataFrame in memory

import os
import time

import pandas as pd
import requests

# --- Settings ---
INPUT_CSV      = "meeting_data.csv"
OUTPUT_CSV     = "meeting_data.csv"
BACKEND        = "gemini"
GEMINI_MODEL   = "gemini-2.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_MODEL   = "llama3"
# ----------------

# CORRECTION_PROMPT = (
#     "You are a transcript editor. "
#     "Fix ONLY the spelling, punctuation and capitalisation in the sentence below. "
#     "Do NOT change the meaning, do NOT add new words, do NOT remove words. "
#     "Return ONLY the corrected sentence — no explanation, no quotes.\n\n"
#     "Sentence: {raw}"
# )

CORRECTION_PROMPT = (
    "You are correcting a speech-to-text transcript from a meeting. "
    "The audio was transcribed automatically and may contain misheared words, "
    "wrong words, or phrases that don't make sense. "
    "Fix ALL errors including wrong/misheared words, grammar, word order, "
    "spelling, punctuation and capitalisation. "
    "Use context to figure out what the speaker most likely meant. "
    "Return ONLY the corrected sentence — no explanation, no quotes.\n\n"
    "Sentence: {raw}"
)


def correct_with_gemini(raw_text):
    import time
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=CORRECTION_PROMPT.format(raw=raw_text)
            )
            return response.text.strip()
        except Exception as e:
            if attempt < 2:
                print(f"Gemini busy, retrying in 10s... (attempt {attempt+1}/3)")
                time.sleep(10)
            else:
                raise e


def correct_with_ollama(raw_text):
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": CORRECTION_PROMPT.format(raw=raw_text),
        "stream": False,
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["response"].strip()


def correct_one_row(raw_text):
    if not raw_text or raw_text.strip() in ("", "[nothing detected]", "[no speech detected]"):
        return raw_text
    if BACKEND == "gemini":
        return correct_with_gemini(raw_text)
    elif BACKEND == "ollama":
        return correct_with_ollama(raw_text)
    else:
        raise ValueError(f"BACKEND must be 'gemini' or 'ollama', got '{BACKEND}'")


def main():
    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    if "raw_text_vosk" not in df.columns:
        raise ValueError("Column 'raw_text_vosk' not found. Run record_audio.py first.")

    if "text" not in df.columns:
        df["text"] = ""

    print(f"Correcting {len(df)} rows using {BACKEND}...\n")

    for idx, row in df.iterrows():
        raw = str(row["raw_text_vosk"])
        print(f"  Row {idx+1}/{len(df)} | {row.get('name','?')}")
        print(f"    Before: {raw}")
        corrected = correct_one_row(raw)
        df.at[idx, "text"] = corrected
        print(f"    After:  {corrected}\n")
        time.sleep(0.3)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done! Saved to '{OUTPUT_CSV}'")


if __name__ == "__main__":
    main()
