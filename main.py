from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from openai import AsyncOpenAI
import os
import uuid
from urllib.parse import urlparse
from dotenv import load_dotenv
import torch
import subprocess
import whisper
import logging
import json
from fastapi.concurrency import run_in_threadpool

# Load environment variables
load_dotenv()

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "medium")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Whisper
logger.info(f"--- Loading Whisper model: {WHISPER_MODEL_NAME}")
device = "cuda" if torch.cuda.is_available() else "cpu"
asr_model = whisper.load_model(WHISPER_MODEL_NAME, device=device)
logger.info(f"--- Whisper model loaded on {device}")

client = AsyncOpenAI()

# Updated prompt with proper guidance
system_msg = """
You are a clinical assistant that extracts structured medical prescription information from a doctor-patient consultation transcript.

The transcript may be in Hindi or English, but your output should always be in English JSON format.

Extract the following:
- chief_complaints (headache, fever, cold, etc)
- diagnose (high BP, low BP, diabetes, etc)
- lab (blood test, urine test, CT scan, etc)
- instructions (avoid spicy food, get enough sleep, etc)
- follow_up (in days : eg : 5 days, 10 days, 15 days) 
- drug_allergies (penicillin, etc)
- medications (list of medicine objects)

Each medication object should have:
- medicine_name (name of the medicine)
- dosage (10mg/20mg/5 ml/10 ml/etc)
- frequency (morning/evening/night/SOS)
- duration (3 days/1 week/2 weeks/1 month)  
- instructions (take medicine on empty stomach, etc)
- medicine_type (Tablet/Injection/Cream/etc)

Always return "medications" as a list, even if only one is present.

Use only the brand names mentioned. If uncertain, return "unknown".

Here are common Indian brand names to help guide recognition:
[
  "Aciloc", "Alex", "Allegra", "Amlopres", "Ascoril", "Aten", "Augmentin", "Avil", "Azithral", 
  "Becosules", "Benadryl", "Betnovate", 
  "Calpol", "Candid", "Cetcip", "Chymoral Forte", "Ciplox", "Clocip", "Cofsils", "Corex DX", "Crocin", "Crocin Advance", "Cyclopam", 
  "D-Rise", "Dermi 5", "Dexorange", "Digene", "Dolo 650", "Dolonex", "Domstal", "Doxy 1", 
  "Etoshine", "Evion", 
  "Fefol", "Flexon", 
  "Galvus Met", "Gelusil", "Gluconorm", "Glycomet", "Grilinctus", 
  "Headset", "Honitus", 
  "Janumet", 
  "L Hist", "LCZ", "Levocet-M", "Livogen", "Lobate", "Losar", 
  "Meftal Spas", "Metosartan", "Metrogyl", "Mixtard", "Montair", "Moxikind-CV", "Mupimet", "Muscoril", "Myoril", 
  "Naxdom", "Nebicard", "Neurobion Forte", "Norflox", "Novorapid", "Nucoxia", 
  "Oflox", "Olvance", "Omez", "Ondem", 
  "Pan D", "Panderm", 
  "Quadriderm", 
  "Rabekind D", "Razo D", 
  "Saridon", "Shelcal", "Sinarest", "Skelebenz", "Stamlo", "Suminat", "Supradyn", 
  "T-Minic", "Taxim-O", "Tazloc", "Telekast", "Telma", "Telmikind", "Tenovate", "Thiocolchicoside", "Trajenta", "Tresiba", "Tryptomer", "TusQ", 
  "Voglibose MD", "Voveran", 
  "Zempred", "Zerodol", "Zifi", "Zincovit", "Zole F", "Zoryl", "Zyrtec"
]

Respond with a single JSON object only.
"""

class AudioRequest(BaseModel):
    audio_url: str

@app.post("/transcribe")
async def transcribe_audio(request: AudioRequest):
    raw_path = wav_path = None
    try:
        def download_audio():
            try:
                response = requests.get(request.audio_url, timeout=10)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                logger.error(f"Download error: {e}")
                raise HTTPException(status_code=400, detail=f"Audio download failed: {e}")

        response = await run_in_threadpool(download_audio)

        parsed = urlparse(request.audio_url)
        _, ext = os.path.splitext(parsed.path)
        raw_path = f"temp_{uuid.uuid4()}{ext}"
        wav_path = raw_path.replace(ext, ".wav")

        with open(raw_path, "wb") as f:
            f.write(response.content)

        subprocess.run([
            "ffmpeg", "-y", "-i", raw_path,
            "-ac", "1", "-ar", "16000", wav_path
        ], check=True)

        result = asr_model.transcribe(wav_path, language="hi", fp16=torch.cuda.is_available())
        transcript = result["text"]
        logger.info(f"Transcript: {transcript}")

        chat_response = await client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": transcript}
            ],
            temperature=0.1
        )

        raw_content = chat_response.choices[0].message.content.strip()

        try:
            structured_output = json.loads(raw_content)
        except json.JSONDecodeError:
            structured_output = {"raw_output": raw_content}
            logger.warning("Could not parse response as JSON.")

        usage = chat_response.usage
        cost_input = (usage.prompt_tokens / 1000) * 0.0011
        cost_output = (usage.completion_tokens / 1000) * 0.0044
        total_cost = cost_input + cost_output

        return {
            "structured_output": structured_output,
            "token_usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            },
            "estimated_cost_usd": round(total_cost, 6)
        }

    except HTTPException:
        raise
    except subprocess.CalledProcessError as cpe:
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {cpe}")
    except Exception as e:
        logger.exception("Unexpected error:")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for f in [raw_path, wav_path]:
            if f and os.path.exists(f):
                os.remove(f)
