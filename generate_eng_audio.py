import os
import sys
import shutil
import json
import asyncio
import edge_tts
from gtts import gTTS

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

AUDIO_DIR = "audio"
TEMP_DIR = "temp_old_eng_audio"
QUESTIONS_FILE = "questions.json"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# Step 1: Move existing English audio files to temp_old_eng_audio
moved_count = 0
for f in os.listdir(AUDIO_DIR):
    if f.startswith("q") and f.endswith("_eng.wav"):
        src_path = os.path.join(AUDIO_DIR, f)
        dst_path = os.path.join(TEMP_DIR, f)
        shutil.move(src_path, dst_path)
        moved_count += 1

print(f"Moved {moved_count} old English audio files to '{TEMP_DIR}/'")

# Step 2: Delete/clean any remaining audio files in audio/
deleted_count = 0
for f in os.listdir(AUDIO_DIR):
    if f.endswith(".wav") or f.endswith(".mp3"):
        os.remove(os.path.join(AUDIO_DIR, f))
        deleted_count += 1

if deleted_count > 0:
    print(f"Deleted {deleted_count} remaining old audio files from '{AUDIO_DIR}/'")

# Step 3: Ensure temp_old_eng_audio is in .gitignore
gitignore_path = ".gitignore"
if os.path.exists(gitignore_path):
    with open(gitignore_path, "r", encoding="utf-8") as gf:
        lines = gf.read().splitlines()
    if TEMP_DIR not in lines and f"{TEMP_DIR}/" not in lines:
        with open(gitignore_path, "a", encoding="utf-8") as gf:
            gf.write(f"\n{TEMP_DIR}/\n")
        print(f"Added '{TEMP_DIR}/' to .gitignore")

# Step 4: Load questions from questions.json
if not os.path.exists(QUESTIONS_FILE):
    print(f"Error: {QUESTIONS_FILE} not found!")
    sys.exit(1)

with open(QUESTIONS_FILE, "r", encoding="utf-8") as qf:
    questions = json.load(qf)

print(f"Loaded {len(questions)} questions from '{QUESTIONS_FILE}'")

ENG_VOICE = "en-IN-PrabhatNeural"

async def generate_all():
    print(f"\nSynthesizing {len(questions)} English male audio files using '{ENG_VOICE}'...")
    for idx, q in enumerate(questions, start=1):
        filepath = os.path.join(AUDIO_DIR, f"q{idx}_eng.wav")
        text = f"Question number {idx}. {q['question']}"
        print(f"[{idx}/{len(questions)}] Synthesizing English: {filepath}...")
        communicate = edge_tts.Communicate(text, ENG_VOICE, rate="-3%")
        await communicate.save(filepath)

    print(f"\nSynthesizing {len(questions)} Malayalam smooth audio files using Google TTS (gTTS)...")
    for idx, q in enumerate(questions, start=1):
        filepath = os.path.join(AUDIO_DIR, f"q{idx}_mal.wav")
        text = f"ചോദ്യം {idx}. {q['malayalam_question']}"
        print(f"[{idx}/{len(questions)}] Synthesizing Malayalam: {filepath}...")
        tts = gTTS(text=text, lang='ml')
        tts.save(filepath)

    print(f"\nAll {len(questions) * 2} audio files (Smooth English & Malayalam voices) successfully generated!")

if __name__ == "__main__":
    asyncio.run(generate_all())
