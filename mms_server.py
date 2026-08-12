import sys
import io
import os
import scipy.io.wavfile
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from transformers import VitsModel, AutoTokenizer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

MODEL_ID = "facebook/mms-tts-mal"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
PORT = 5000

print(f"Loading {MODEL_ID} model and tokenizer...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = VitsModel.from_pretrained(MODEL_ID, token=HF_TOKEN).to(device)
    print("✅ Facebook MMS-TTS Malayalam model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

audio_cache = {}

class TTSRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress routine log output for cleaner console
        return

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok", "model": "facebook/mms-tts-mal"}')
            return

        if parsed.path == "/tts":
            params = parse_qs(parsed.query)
            text = params.get("text", [""])[0]
            if not text:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing text parameter")
                return

            if text in audio_cache:
                print(f"⚡ Serving cached audio for: {text[:60]}...")
                wav_data = audio_cache[text]
            else:
                print(f"🔊 Synthesizing: {text[:60]}...")
                try:
                    inputs = tokenizer(text, return_tensors="pt").to(device)
                    with torch.no_grad():
                        output = model(**inputs).waveform
                        # Normalize to peak volume (98%) for loud & clear audio
                        max_val = torch.max(torch.abs(output))
                        if max_val > 0:
                            output = (output / max_val) * 0.98

                    audio_np = output.squeeze().cpu().numpy()
                    sampling_rate = model.config.sampling_rate

                    wav_buffer = io.BytesIO()
                    scipy.io.wavfile.write(wav_buffer, rate=sampling_rate, data=audio_np)
                    wav_data = wav_buffer.getvalue()
                    audio_cache[text] = wav_data
                    print("✨ Audio generated & cached.")
                except Exception as e:
                    print(f"❌ TTS Error: {e}")
                    self.send_response(500)
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(str(e).encode("utf-8"))
                    return

            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav_data)))
            self.end_headers()
            self.wfile.write(wav_data)
            return

        self.send_response(404)
        self.end_headers()

def run_server():
    server = HTTPServer(("0.0.0.0", PORT), TTSRequestHandler)
    print(f"\n🚀 Facebook MMS-TTS Malayalam Server running on http://localhost:{PORT}")
    print("Ready to serve TTS requests for madrass.html!")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()

if __name__ == "__main__":
    run_server()
