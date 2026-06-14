# Meeting Speech Analytics — Vosk + Gemini Pipeline

Records team meeting speech, transcribes with Vosk, corrects with Gemini AI,
enriches the dataset, validates it, and produces speaking analytics.

## Team
| Name    | What they built                                                   |
|---------|----------------------------------------                           |
| Flora   | Stage  1 — record_audio.py                                        |
| Flora   | Stage  2 - AI transcript correction                               |
| Mustafa |  Stage 3 — enrich_data.py                                         |
| Tracy   | Stage  4 - validate.py, analyse.py                                |
| Kamran  | Stage  5 — README, complexity docs,                               |

## Setup

```
pip install -r requirements.txt 
export GEMINI_API_KEY="your-key-here"
```

Download Vosk model from https://alphacephei.com/vosk/models
Unzip so folder contains: vosk-model-en-us-0.22-lgraph/

## How to run — in order, in terminal

```
python record_audio.py
python correct_transcripts.py
python enrich_data.py
python validate.py
python analyse.py
```

## Final CSV columns

| Column           | How it is calculated                        |
|------------------|---------------------------------------------|
| timestamp        | Date and time of recording                  |
| name             | Speaker name                                |
| raw_text_vosk    | Raw Vosk output (lowercase, no punctuation) |
| text             | AI-corrected version                        |
| time_taken_sec   | Recording duration in seconds               |
| question_flag    | True if text ends with ?                    |
| num_words        | Word count of corrected text                |
| text_size_chars  | Character count of corrected text           |
| speech_rate_wps  | num_words divided by time_taken_sec         |
| speaker_turn_id  | Per-speaker turn number (1, 2, 3...)        |

## Time and Space Complexity

### record_audio.py
- Model loading: O(1) time, O(M) space (~128 MB)
- Per recording: O(N) time and space (N = audio samples)
- CSV append: O(1) time and space

### correct_transcripts.py
- Per API call: O(T) time and space (T = tokens)
- All rows: O(N x T) time, O(N) space

### enrich_data.py
- question_flag, num_words, text_size_chars, speech_rate_wps: O(N) time, O(N) space
- speaker_turn_id: O(N log N) time, O(N) space

### validate.py
- O(N x C) time, O(N) space (C = columns checked)

### analyse.py
- O(N log N) time, O(N + S) space (S = unique speakers)

## Known Limitations

- Vosk isn't perfect — it mishears words in casual conversation, 
  so we used Gemini to clean up the transcripts afterwards.
- Gemini can't rehear the audio, so some phrases were still wrong 
  and needed to be fixed manually.
- Speaker names are automatically tidied up since they were sometimes 
  typed inconsistently (e.g. "kamran" vs "Kamran").