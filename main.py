from fastapi import FastAPI
from pydantic import BaseModel
from rag_context import ask_llm_interpreter

app = FastAPI()

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Question):
    answer = ask_llm_interpreter(q.question)
    return {"answer": answer}
