import os
import sys
import json
import scipy.io.wavfile
import torch
from transformers import VitsModel, AutoTokenizer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

OUTPUT_DIR = "audio"
QUESTIONS_FILE = "questions.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if not os.path.exists(QUESTIONS_FILE):
    print(f"Error: {QUESTIONS_FILE} not found!")
    sys.exit(1)

with open(QUESTIONS_FILE, "r", encoding="utf-8") as qf:
    questions = json.load(qf)

print(f"Loaded {len(questions)} questions from '{QUESTIONS_FILE}'")

print("Loading Malayalam TTS model (facebook/mms-tts-mal)...")
tok_mal = AutoTokenizer.from_pretrained("facebook/mms-tts-mal", token=HF_TOKEN)
mod_mal = VitsModel.from_pretrained("facebook/mms-tts-mal", token=HF_TOKEN).to(device)

print("Loading English TTS model (facebook/mms-tts-eng)...")
tok_eng = AutoTokenizer.from_pretrained("facebook/mms-tts-eng", token=HF_TOKEN)
mod_eng = VitsModel.from_pretrained("facebook/mms-tts-eng", token=HF_TOKEN).to(device)

def generate_wav(text, tokenizer, model, filepath):
    try:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs).waveform
            max_val = torch.max(torch.abs(output))
            if max_val > 0:
                output = (output / max_val) * 0.98
        audio_np = output.squeeze().cpu().numpy()
        sr = model.config.sampling_rate
        scipy.io.wavfile.write(filepath, rate=sr, data=audio_np)
        print(f"  ✅ Saved {filepath}")
    except Exception as e:
        print(f"  ❌ Error generating {filepath}: {e}")
        import traceback
        traceback.print_exc()

print(f"\nGenerating {len(questions) * 2} audio files into '{OUTPUT_DIR}'...")

for idx, q in enumerate(questions, start=1):
    eng_file = os.path.join(OUTPUT_DIR, f"q{idx}_eng.wav")
    mal_file = os.path.join(OUTPUT_DIR, f"q{idx}_mal.wav")
    
    eng_text = f"Question number {idx}. {q['question']}"
    mal_text = f"ചോദ്യം {idx}. {q['malayalam_question']}"
    
    print(f"[{idx}/{len(questions)}] Generating English audio: {eng_file}...")
    generate_wav(eng_text, tok_eng, mod_eng, eng_file)
    
    print(f"[{idx}/{len(questions)}] Generating Malayalam audio: {mal_file}...")
    generate_wav(mal_text, tok_mal, mod_mal, mal_file)

print("\n🎉 Process completed! Check 'audio/' folder for generated files.")
