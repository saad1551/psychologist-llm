from fastapi import FastAPI
from llm import get_response

app = FastAPI()

@app.post("/llm")
def llm(message: str):
    reply = get_response(message)

    return {
        "message": reply
    }