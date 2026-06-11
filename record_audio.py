# record_audio.py
# ---------------
# Stage 1 of the pipeline.
# Records audio from the microphone using Vosk and saves each turn
# as a row in meeting_data.csv.
#
# HOW TO RUN:
#   - Open your terminal
#   - Navigate to this folder: cd path/to/meeting-project
#   - Run: python record_audio.py
#
# BEFORE RUNNING:
#   1. Download the Vosk model from https://alphacephei.com/vosk/models
#      (download vosk-model-en-us-0.22-lgraph)
#   2. Unzip it so your folder contains: vosk-model-en-us-0.22-lgraph/
#
# Time complexity:
#   Model loading: O(1) — one-time read at startup
#   Per recording turn: O(N) — N = audio samples
#   CSV append: O(1) — one row per turn
#
# Space complexity:
#   Model in memory: O(M) — about 128 MB
#   Audio buffer: O(N) per turn, then discarded

import csv
import json
import os
import queue
import sys
import time
from datetime import datetime

import pandas as pd
import sounddevice as sd
from vosk import KaldiRecognizer, Model

# --- Settings ---
MODEL_PATH  = "vosk-model-en-us-0.22-lgraph"
OUTPUT_CSV  = "meeting_data.csv"
SAMPLE_RATE = 16000
RECORD_SECS = 10
CSV_COLUMNS = ["timestamp", "name", "raw_text_vosk", "time_taken_sec"]
# ----------------


def load_model(model_path):
    if not os.path.exists(model_path):
        sys.exit(
            f"\nCould not find the Vosk model folder: '{model_path}'\n"
            "Download it from https://alphacephei.com/vosk/models\n"
            "Unzip it so the folder sits inside your project directory.\n"
        )
    print(f"Loading Vosk model from '{model_path}', this takes a few seconds...")
    return Model(model_path)


def record_and_transcribe(model, duration):
    audio_q = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"Warning: {status}", file=sys.stderr)
        audio_q.put(bytes(indata))

    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    transcript = ""
    start = time.time()

    print(f"  Recording for {duration} seconds — speak now!")

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        deadline = start + duration
        while time.time() < deadline:
            chunk = audio_q.get()
            if recognizer.AcceptWaveform(chunk):
                result = json.loads(recognizer.Result())
                transcript += " " + result.get("text", "")

        final = json.loads(recognizer.FinalResult())
        transcript += " " + final.get("text", "")

    elapsed = round(time.time() - start, 2)
    return transcript.strip(), elapsed


def append_to_csv(filepath, row):
    new_file = not os.path.isfile(filepath)
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
            print(f"Created new file: {filepath}")
        writer.writerow(row)


def record_turn(model, name):
    print(f"\n--- Now recording: {name} ---")
    raw_text, elapsed = record_and_transcribe(model, RECORD_SECS)

    row = {
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
        "name":           name,
        "raw_text_vosk":  raw_text if raw_text else "[nothing detected]",
        "time_taken_sec": elapsed,
    }
    append_to_csv(OUTPUT_CSV, row)
    print(f"  Saved: '{row['raw_text_vosk']}' ({elapsed}s)")


def main():
    model = load_model(MODEL_PATH)

    print("\n========================================")
    print("  Meeting Recorder — powered by Vosk")
    print("========================================")
    print(f"  Saving to  : {OUTPUT_CSV}")
    print(f"  Time limit : {RECORD_SECS}s per person")
    print("  Type a name and press Enter to start recording.")
    print("  Type 'done' when finished.\n")

    while True:
        name = input("Speaker name (or 'done' to finish): ").strip()
        if name.lower() == "done":
            break
        if not name:
            print("Please enter a name before pressing Enter.")
            continue
        record_turn(model, name)

    if os.path.isfile(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        print(f"\nDone! {len(df)} rows saved to '{OUTPUT_CSV}'")
        print(df.to_string(index=False))
    else:
        print("No rows were recorded.")


if __name__ == "__main__":
    main()
