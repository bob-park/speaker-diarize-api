# Speaker Diarization with PyTorch
이것은 오디오 파일에서 화자를 분리하여 추출하는 기능이 포함된 API 이다.



## Docker 실행
```bash
docker run -it -d \
  --gpus all \
  -p 5001:5001 \
  -e HUGGINGFACE_ACCESS_TOKEN={accessToken} \
  --name speaker-diarize-api \
  ghcr.io/bob-park/speaker-diarize-api
```
