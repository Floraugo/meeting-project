# enrich_data.py
# --------------
# Stage 3 of the pipeline.
# Adds calculated columns using Python only — no AI involved here.
#
# HOW TO RUN:
#   python enrich_data.py
#
# Columns added:
#   question_flag    — True if corrected text ends with ?
#   num_words        — word count of corrected text
#   text_size_chars  — character count of corrected text
#   speech_rate_wps  — words per second
#   speaker_turn_id  — per-speaker turn count (1, 2, 3...)
#
# Time complexity:
#   question_flag, num_words, text_size_chars, speech_rate_wps: O(N)
#   speaker_turn_id: O(N log N) — pandas groupby sorts internally
#
# Space complexity: O(N) — DataFrame in memory

import pandas as pd

INPUT_CSV  = "meeting_data.csv"
OUTPUT_CSV = "meeting_data_enriched.csv"


def add_question_flag(df):
    df["question_flag"] = df["text"].apply(lambda t: str(t).strip().endswith("?"))
    return df


def add_num_words(df):
    df["num_words"] = df["text"].apply(lambda t: len(str(t).split()))
    return df


def add_text_size_chars(df):
    df["text_size_chars"] = df["text"].apply(lambda t: len(str(t)))
    return df


def add_speech_rate_wps(df):
    def calculate_rate(row):
        seconds = float(row["time_taken_sec"])
        return round(row["num_words"] / seconds, 2) if seconds > 0 else 0.0
    df["speech_rate_wps"] = df.apply(calculate_rate, axis=1)
    return df


def add_speaker_turn_id(df):
    df["speaker_turn_id"] = df.groupby("name").cumcount() + 1
    return df


def enrich(input_path, output_path):
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    df["name"] = df["name"].str.strip().str.title()

    required_cols = {"text", "time_taken_sec", "name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            "Run correct_transcripts.py before enrich_data.py."
        )

    print("Adding enrichment columns...")
    df = add_question_flag(df)
    df = add_num_words(df)
    df = add_text_size_chars(df)
    df = add_speech_rate_wps(df)
    df = add_speaker_turn_id(df)

    df.to_csv(output_path, index=False)
    print(f"\nSaved to '{output_path}' — {len(df)} rows\n")

    preview_cols = [
        "timestamp", "name", "text", "time_taken_sec",
        "question_flag", "num_words", "text_size_chars",
        "speech_rate_wps", "speaker_turn_id"
    ]
    print(df[preview_cols].to_string(index=False))
    return df


if __name__ == "__main__":
    enrich(INPUT_CSV, OUTPUT_CSV)
