# validate.py
# -----------
# Stage 4 of the pipeline.
# Checks the enriched CSV for missing values, wrong types,
# and minimum row count before running analytics.
#
# HOW TO RUN:
#   python validate.py
#
# You want to see "Validation PASSED" before running analyse.py.
#
# Time complexity:  O(N x C) — N rows, C columns checked
# Space complexity: O(N) — DataFrame in memory

import sys
import pandas as pd

INPUT_CSV        = "meeting_data_enriched.csv"
MIN_ROWS         = 25

REQUIRED_COLUMNS = [
    "timestamp", "name", "raw_text_vosk", "text",
    "time_taken_sec", "question_flag", "num_words",
    "text_size_chars", "speech_rate_wps", "speaker_turn_id",
]


def validate(filepath):
    errors   = []
    warnings = []

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Could not find '{filepath}'. Run enrich_data.py first.")
        return False

    print(f"Loaded '{filepath}': {len(df)} rows, {len(df.columns)} columns\n")

    if len(df) < MIN_ROWS:
        errors.append(f"Only {len(df)} rows — need at least {MIN_ROWS}.")

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing column: '{col}'")

    if errors:
        report(errors, warnings)
        return False

    for col in REQUIRED_COLUMNS:
        blank_rows = df.index[df[col].isnull()].tolist()
        for i in blank_rows:
            errors.append(f"Row {i + 2}: '{col}' is empty.")

    for i, val in enumerate(df["timestamp"]):
        try:
            pd.to_datetime(val)
        except Exception:
            errors.append(f"Row {i + 2}: timestamp '{val}' is not valid.")

    for col in ("time_taken_sec", "num_words", "speech_rate_wps", "speaker_turn_id"):
        converted = pd.to_numeric(df[col], errors="coerce")
        for i in df.index[converted.isna()]:
            errors.append(f"Row {i + 2}: '{col}' is not a number.")
        for i in df.index[converted.fillna(0) <= 0]:
            if not converted.isna()[i]:
                errors.append(f"Row {i + 2}: '{col}' must be greater than 0.")

    valid_bool_values = {True, False, "True", "False", 1, 0, "true", "false"}
    for i, val in enumerate(df["question_flag"]):
        if val not in valid_bool_values:
            errors.append(f"Row {i + 2}: 'question_flag' must be True or False (got: {val!r}).")

    for i, val in enumerate(df["name"]):
        if not str(val).strip():
            warnings.append(f"Row {i + 2}: speaker name is blank.")

    report(errors, warnings)
    return len(errors) == 0


def report(errors, warnings):
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  ! {w}")
        print()

    if errors:
        print("Validation FAILED — fix these before running analyse.py:")
        for e in errors:
            print(f"  x {e}")
    else:
        print("Validation PASSED — data looks good, run analyse.py next!")


if __name__ == "__main__":
    ok = validate(INPUT_CSV)
    sys.exit(0 if ok else 1)
