from fastapi import FastAPI
from llm import generate_reply

app = FastAPI()

@app.post("/llm")
def llm(message: str):
    reply = generate_reply(message)

    return {
        "message": reply
    }