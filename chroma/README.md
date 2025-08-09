# Chroma
windows powershell에서 테스트함

cuda : 12.8

## upload
- pyproject.toml : 의존성+패키지데이터

- requirements.txt : 패키지 리스트

- rag_chroma.py : 실행파일


## uv환경구축
```
uv venv
.venv/Scripts/activate
uv sync
uv pip install -r requirements.txt
```

requirements.txt 파일 내에 torch, torchvision, torchaudio의 cuda버전은 상황에 맞게 수정
```
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
