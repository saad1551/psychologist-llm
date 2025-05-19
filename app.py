from fastapi import FastAPI
from llm import get_response
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/llm")
def llm(message: str):
    reply = get_response(message)

    return {
        "message": reply
    }