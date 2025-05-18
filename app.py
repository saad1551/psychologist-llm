from fastapi import FastAPI

app = FastAPI()

@app.post("/llm")
def llm(message: str):
    return 