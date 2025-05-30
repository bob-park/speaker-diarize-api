from flask import Flask, request, jsonify
from pyannote.audio import Pipeline
import os
import tempfile
import torch

app = Flask(__name__)

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token=os.environ["HUGGINGFACE_ACCESS_TOKEN"])

# pipeline.to(torch.device("cpu"))
pipeline.to(torch.device("cuda"))

@app.route("/diarize", methods=["POST"])
def diarize():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    audio_file = request.files['file']

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_path = tmp.name
        audio_file.save(audio_path)

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
    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
