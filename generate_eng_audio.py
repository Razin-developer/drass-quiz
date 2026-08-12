import os
import sys
import shutil
import asyncio
import edge_tts

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Step 1: Move existing English audio files to temp_old_eng_audio
AUDIO_DIR = "audio"
TEMP_DIR = "temp_old_eng_audio"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

moved_count = 0
for f in os.listdir(AUDIO_DIR):
    if f.startswith("q") and f.endswith("_eng.wav"):
        src_path = os.path.join(AUDIO_DIR, f)
        dst_path = os.path.join(TEMP_DIR, f)
        shutil.move(src_path, dst_path)
        moved_count += 1

print(f"Moved {moved_count} old English audio files to '{TEMP_DIR}/'")

# Step 2: Ensure temp_old_eng_audio is in .gitignore
gitignore_path = ".gitignore"
if os.path.exists(gitignore_path):
    with open(gitignore_path, "r", encoding="utf-8") as gf:
        lines = gf.read().splitlines()
    if TEMP_DIR not in lines:
        with open(gitignore_path, "a", encoding="utf-8") as gf:
            gf.write(f"\n{TEMP_DIR}/\n")
        print(f"Added '{TEMP_DIR}/' to .gitignore")

# Step 3: Questions list
questions = [
    {
        "id": 1,
        "qn_eng": "Who was the first Prime Minister of independent India to hoist the national flag at the Red Fort on August 15, 1947?"
    },
    {
        "id": 2,
        "qn_eng": "Which freedom fighter designed the original Swaraj Flag in 1921?"
    },
    {
        "id": 3,
        "qn_eng": "Who gave the famous slogan 'Give me blood, and I shall give you freedom'?"
    },
    {
        "id": 4,
        "qn_eng": "In which year was the Quit India Movement launched by Mahatma Gandhi?"
    },
    {
        "id": 5,
        "qn_eng": "Who was known as the 'Nightingale of India' and actively participated in the freedom struggle?"
    },
    {
        "id": 6,
        "qn_eng": "Which movement was started by Gandhi in 1930 after breaking the salt law at Dandi?"
    },
    {
        "id": 7,
        "qn_eng": "Who wrote the national anthem of India, 'Jana Gana Mana'?"
    },
    {
        "id": 8,
        "qn_eng": "Who was the leader of the Bardoli Satyagraha and earned the title 'Sardar'?"
    },
    {
        "id": 9,
        "qn_eng": "Who reorganized and led the Indian National Army (INA)?"
    },
    {
        "id": 10,
        "qn_eng": "What is the ratio of the width of the Indian National Flag to its length?"
    },
    {
        "id": 11,
        "qn_eng": "Who was the British Viceroy of India during the Partition in 1947?"
    },
    {
        "id": 12,
        "qn_eng": "Which martyr said 'Inquilab Zindabad' in the Central Assembly in 1929?"
    },
    {
        "id": 13,
        "qn_eng": "In which city did the tragic Jallianwala Bagh massacre take place in 1919?"
    },
    {
        "id": 14,
        "qn_eng": "Who was the founder of the Servants of India Society and political guru of Mahatma Gandhi?"
    },
    {
        "id": 15,
        "qn_eng": "Which freedom fighter was popularly known as 'Frontier Gandhi'?"
    },
    {
        "id": 16,
        "qn_eng": "Who wrote the national song of India, 'Vande Mataram'?"
    },
    {
        "id": 17,
        "qn_eng": "Who was the first President of Independent India?"
    },
    {
        "id": 18,
        "qn_eng": "Who was known as the 'Iron Man of India'?"
    },
    {
        "id": 19,
        "qn_eng": "Where is the National Flag of India hoisted every year on Independence Day by the Prime Minister?"
    },
    {
        "id": 20,
        "qn_eng": "Who was the chairman of the Drafting Committee of the Indian Constitution?"
    }
]

VOICE = "en-IN-PrabhatNeural"

async def generate_all():
    print(f"\nGenerating 20 new natural English male voice files using '{VOICE}'...")
    for q in questions:
        qid = q["id"]
        filepath = os.path.join(AUDIO_DIR, f"q{qid}_eng.wav")
        text = f"Question number {qid}. {q['qn_eng']}"
        print(f"[{qid}/20] Synthesizing: {filepath}...")
        communicate = edge_tts.Communicate(text, VOICE, rate="-3%")
        await communicate.save(filepath)
        print(f"  Saved {filepath}")

    print("\nAll 20 English audio files successfully generated!")

if __name__ == "__main__":
    asyncio.run(generate_all())
