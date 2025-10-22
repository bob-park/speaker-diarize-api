import logging
import os
import tempfile

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pyannote.audio import Pipeline

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()


def getDevice():
    if torch.cuda.is_available():
        logger.debug("CUDA device detected")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logger.debug("MPS device detected (Apple Silicon)")
        return torch.device("mps")  # Apple Silicon GPU
    else:
        logger.debug("Using CPU device")
        return torch.device("cpu")


device = getDevice()

# 환경변수 확인 및 가져오기
hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
if not hf_token:
    logger.error("HUGGINGFACE_ACCESS_TOKEN not found in environment variables")
    logger.debug(f"Available environment variables: {list(os.environ.keys())}")
    raise ValueError("HUGGINGFACE_ACCESS_TOKEN environment variable is required")

logger.debug(f"HuggingFace token loaded (length: {len(hf_token)})")

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token=hf_token)

pipeline.to(device)


@app.get("/health")
async def health():
    return JSONResponse(content={"status": "ok"})


@app.post("/diarize")
async def diarize(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_path = tmp.name
        content = await file.read()
        tmp.write(content)

    # diarization 수행
    diarization = pipeline(audio_path)

    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end
        })

    os.remove(audio_path)  # 임시 파일 정리
    return JSONResponse(content=results)
