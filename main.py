import os
import ollama
from fastapi import FastAPI
from pydantic import BaseModel

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
client = ollama.Client(host=ollama_host)

app = FastAPI()


class Request(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 200


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: Request):
    result = client.chat(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": req.prompt}],
        options={"temperature": req.temperature, "num_predict": req.max_tokens},
    )
    return {"response": result["message"]["content"]}
