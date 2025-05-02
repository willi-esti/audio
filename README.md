```sh
python3 -m venv venv
```

```sh
source venv/bin/activate
pip install websockets pydub openai-whisper ffmpeg-python
```


```sh
docker pull ollama/ollama:latest
docker run -d -p 11434:11434 --name ollama_setup ollama/ollama # Start temporarily
docker exec ollama_setup ollama pull llama3 # Or your chosen model (e.g., mistral)
docker stop ollama_setup && docker rm ollama_setup # Stop and remove temporary container
```

```sh
docker exec ollama ollama pull tinyllama
```


```sh
docker-compose up --build
```







