# analyse.py
# ----------
# Stage 5 — final step of the pipeline.
# Answers all 7 analytics questions from the project brief.
#
# HOW TO RUN:
#   python analyse.py
#
# To save the output to a text file:
#   python analyse.py > analytics_report.txt
#
# Time complexity:  O(N log N) — groupby + sort
# Space complexity: O(N + S) — N rows, S unique speakers

import sys
import pandas as pd

try:
    from tabulate import tabulate
    def make_table(df):
        return tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False)
except ImportError:
    def make_table(df):
        return df.to_string(index=False)

INPUT_CSV = "meeting_data_enriched.csv"


def analyse(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Could not find '{filepath}'. Run enrich_data.py first.")
        sys.exit(1)

    df["num_words"]       = pd.to_numeric(df["num_words"],       errors="coerce").fillna(0)
    df["time_taken_sec"]  = pd.to_numeric(df["time_taken_sec"],  errors="coerce").fillna(0)
    df["speech_rate_wps"] = pd.to_numeric(df["speech_rate_wps"], errors="coerce").fillna(0)
    df["question_flag"]   = (
        df["question_flag"].astype(str).str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
    )

    by_speaker = df.groupby("name").agg(
        total_words    =("num_words",       "sum"),
        total_time_sec =("time_taken_sec",  "sum"),
        avg_rate_wps   =("speech_rate_wps", "mean"),
        question_count =("question_flag",   "sum"),
        turn_count     =("speaker_turn_id", "count"),
    ).reset_index()

    by_speaker["total_time_sec"] = by_speaker["total_time_sec"].round(2)
    by_speaker["avg_rate_wps"]   = by_speaker["avg_rate_wps"].round(2)
    by_speaker["avg_time_sec"]   = (by_speaker["total_time_sec"] / by_speaker["turn_count"]).round(2)

    line = "=" * 58
    print(line)
    print("  MEETING ANALYTICS REPORT")
    print(line)
    print(f"  File    : {filepath}")
    print(f"  Rows    : {len(df)}")
    print(f"  Speakers: {df['name'].nunique()}")
    print(line + "\n")

    most  = by_speaker.loc[by_speaker["total_words"].idxmax()]
    print(f"1. Most words  : {most['name']} with {int(most['total_words'])} words")

    least = by_speaker.loc[by_speaker["total_words"].idxmin()]
    print(f"2. Least words : {least['name']} with {int(least['total_words'])} words\n")

    total_time = round(df["time_taken_sec"].sum(), 2)
    print(f"3. Total speaking time : {total_time} seconds\n")

    avg_table = by_speaker[["name", "avg_time_sec"]].rename(
        columns={"name": "Speaker", "avg_time_sec": "Avg time per turn (sec)"}
    )
    print("4. Average speaking time per speaker:")
    print(make_table(avg_table), "\n")

    most_q = by_speaker.loc[by_speaker["question_count"].idxmax()]
    print(f"5. Most questions : {most_q['name']} with {int(most_q['question_count'])} question(s)\n")

    top5 = (
        by_speaker[["name", "total_time_sec"]]
        .sort_values("total_time_sec", ascending=False)
        .head(5)
        .rename(columns={"name": "Speaker", "total_time_sec": "Total time (sec)"})
        .reset_index(drop=True)
    )
    top5.index += 1
    print("6. Top 5 speakers by total speaking time:")
    print(make_table(top5), "\n")

    rates = (
        by_speaker[["name", "avg_rate_wps"]]
        .sort_values("avg_rate_wps", ascending=False)
        .rename(columns={"name": "Speaker", "avg_rate_wps": "Avg rate (wps)"})
        .reset_index(drop=True)
    )
    print("7. Average speech rate per speaker:")
    print(make_table(rates), "\n")

    print("-" * 58)
    print("FULL SUMMARY")
    print("-" * 58)
    summary = by_speaker.rename(columns={
        "name": "Speaker", "total_words": "Words",
        "total_time_sec": "Time(s)", "avg_rate_wps": "Rate(wps)",
        "question_count": "Questions", "turn_count": "Turns",
        "avg_time_sec": "AvgTime(s)",
    })
    print(make_table(summary[["Speaker","Words","Time(s)","Rate(wps)","Questions","Turns","AvgTime(s)"]]))


if __name__ == "__main__":
    analyse(INPUT_CSV)
