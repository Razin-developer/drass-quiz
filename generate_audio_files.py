import os
import sys
import scipy.io.wavfile
import torch
from transformers import VitsModel, AutoTokenizer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

OUTPUT_DIR = "audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

questions = [
    {
        "id": 1,
        "qn_eng": "Who was the first Prime Minister of independent India to hoist the national flag at the Red Fort on August 15, 1947?",
        "qn_mal": "1947 ഓഗസ്റ്റ് 15-ന് ചെങ്കോട്ടയിൽ ആദ്യമായി ദേശീയ പതാക ഉയർത്തിയ സ്വതന്ത്ര ഇന്ത്യയുടെ ആദ്യ പ്രധാനമന്ത്രി ആര്?"
    },
    {
        "id": 2,
        "qn_eng": "Which freedom fighter designed the original Swaraj Flag in 1921?",
        "qn_mal": "1921-ൽ സ്വരാജ് പതാക രൂപകൽപ്പന ചെയ്ത സ്വാതന്ത്ര്യസമര സേനാനി ആര്?"
    },
    {
        "id": 3,
        "qn_eng": "Who gave the famous slogan 'Give me blood, and I shall give you freedom'?",
        "qn_mal": "'നിങ്ങൾ എനിക്ക് രക്തം തരൂ, ഞാൻ നിങ്ങൾക്ക് സ്വാതന്ത്ര്യം തരാം' എന്ന പ്രശസ്തമായ മുദ്രാവാക്യം ഉയർത്തിയത് ആര്?"
    },
    {
        "id": 4,
        "qn_eng": "In which year was the Quit India Movement launched by Mahatma Gandhi?",
        "qn_mal": "മഹാത്മാഗാന്ധി ക്വിറ്റ് ഇന്ത്യ പ്രസ്ഥാനം ആരംഭിച്ച വർഷം ഏത്?"
    },
    {
        "id": 5,
        "qn_eng": "Who was known as the 'Nightingale of India' and actively participated in the freedom struggle?",
        "qn_mal": "'ഇന്ത്യയുടെ നൈറ്റിംഗേൾ' എന്നറിയപ്പെടുന്ന സ്വാതന്ത്ര്യസമര സേനാനി ആര്?"
    },
    {
        "id": 6,
        "qn_eng": "Which movement was started by Gandhi in 1930 after breaking the salt law at Dandi?",
        "qn_mal": "ദണ്ഡിയിൽ ഉപ്പുനിയമം ലംഘിച്ച് ഗാന്ധിജി 1930-ൽ ആരംഭിച്ച പ്രസ്ഥാനം ഏത്?"
    },
    {
        "id": 7,
        "qn_eng": "Who wrote the national anthem of India, 'Jana Gana Mana'?",
        "qn_mal": "ഇന്ത്യയുടെ ദേശീയ ഗാനമായ 'ജനഗണമന' രചിച്ചത് ആര്?"
    },
    {
        "id": 8,
        "qn_eng": "Who was the leader of the Bardoli Satyagraha and earned the title 'Sardar'?",
        "qn_mal": "ബർദോളി സത്യാഗ്രഹത്തിന് നേതൃത്വം നൽകുകയും 'സർദാർ' എന്ന പദവി ലഭിക്കുകയും ചെയ്തതാര്?"
    },
    {
        "id": 9,
        "qn_eng": "Who reorganized and led the Indian National Army (INA)?",
        "qn_mal": "ഇന്ത്യൻ നാഷണൽ ആർമി (INA) പുനഃസംഘടിപ്പിച്ച് നയിച്ചത് ആര്?"
    },
    {
        "id": 10,
        "qn_eng": "What is the ratio of the width of the Indian National Flag to its length?",
        "qn_mal": "ഇന്ത്യൻ ദേശീയ പതാകയുടെ വീതിയും നീളവും തമ്മിലുള്ള അനുപാതം എത്ര?"
    },
    {
        "id": 11,
        "qn_eng": "Who was the British Viceroy of India during the Partition in 1947?",
        "qn_mal": "1947-ൽ ഇന്ത്യയുടെ വിഭജന സമയത്ത് ബ്രിട്ടീഷ് വൈസ്രോയി ആരായിരുന്നു?"
    },
    {
        "id": 12,
        "qn_eng": "Which martyr said 'Inquilab Zindabad' in the Central Assembly in 1929?",
        "qn_mal": "1929-ൽ സെൻട്രൽ അസംബ്ലിയിൽ 'ഇൻക്വിലാബ് സിന്ദാബാദ്' എന്ന മുദ്രാവാക്യം മുഴക്കിയ രക്തസാക്ഷി ആര്?"
    },
    {
        "id": 13,
        "qn_eng": "In which city did the tragic Jallianwala Bagh massacre take place in 1919?",
        "qn_mal": "1919-ൽ ദാരുണമായ ജാലിയൻവാലാബാഗ് കൂട്ടക്കൊല നടന്ന നഗരം ഏത്?"
    },
    {
        "id": 14,
        "qn_eng": "Who was the founder of the Servants of India Society and political guru of Mahatma Gandhi?",
        "qn_mal": "സെർവന്റ്സ് ഓഫ് ഇന്ത്യ സൊസൈറ്റിയുടെ സ്ഥാപകനും ഗാന്ധിജിയുടെ രാഷ്ട്രീയ ഗുരുവും ആര്?"
    },
    {
        "id": 15,
        "qn_eng": "Which freedom fighter was popularly known as 'Frontier Gandhi'?",
        "qn_mal": "'അതിർത്തി ഗാന്ധി' എന്നറിയപ്പെട്ടിരുന്ന സ്വാതന്ത്ര്യസമര സേനാനി ആര്?"
    },
    {
        "id": 16,
        "qn_eng": "Who wrote the national song of India, 'Vande Mataram'?",
        "qn_mal": "ഇന്ത്യയുടെ ദേശീയ ഗീതമായ 'വന്ദേ മാതരം' രചിച്ചത് ആര്?"
    },
    {
        "id": 17,
        "qn_eng": "Who was the first President of Independent India?",
        "qn_mal": "സ്വതന്ത്ര ഇന്ത്യയുടെ ആദ്യ പ്രസിഡന്റ് ആരായിരുന്നു?"
    },
    {
        "id": 18,
        "qn_eng": "Who was known as the 'Iron Man of India'?",
        "qn_mal": "'ഇന്ത്യയുടെ ഇരുമ്പ് മനുഷ്യൻ' എന്നറിയപ്പെടുന്നത് ആര്?"
    },
    {
        "id": 19,
        "qn_eng": "Where is the National Flag of India hoisted every year on Independence Day by the Prime Minister?",
        "qn_mal": "സ്വാതന്ത്ര്യദിനത്തിൽ ഓരോ വർഷവും പ്രധാനമന്ത്രി ദേശീയ പതാക ഉയർത്തുന്നത് എവിടെയാണ്?"
    },
    {
        "id": 20,
        "qn_eng": "Who was the chairman of the Drafting Committee of the Indian Constitution?",
        "qn_mal": "ഇന്ത്യൻ ഭരണഘടനയുടെ ഡ്രാഫ്റ്റിംഗ് കമ്മിറ്റിയുടെ ചെയർമാൻ ആരായിരുന്നു?"
    }
]

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

print(f"\nGenerating 40 audio files (20 Malayalam + 20 English) into '{OUTPUT_DIR}'...")

for q in questions:
    qid = q["id"]
    eng_file = os.path.join(OUTPUT_DIR, f"q{qid}_eng.wav")
    mal_file = os.path.join(OUTPUT_DIR, f"q{qid}_mal.wav")
    
    eng_text = f"Question number {qid}. {q['qn_eng']}"
    mal_text = f"ചോദ്യം {qid}. {q['qn_mal']}"
    
    print(f"[{qid}/20] Generating English audio: {eng_file}...")
    generate_wav(eng_text, tok_eng, mod_eng, eng_file)
    
    print(f"[{qid}/20] Generating Malayalam audio: {mal_file}...")
    generate_wav(mal_text, tok_mal, mod_mal, mal_file)

print("\n🎉 Process completed! Check 'audio/' folder for generated files.")
