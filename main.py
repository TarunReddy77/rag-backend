from typing import List
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from utils.file_parser import extract_text_from_pdf
from utils.rag_engine import load_text_into_vectorstore, ask_question
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "RAG backend is running"}


# Temporary in-memory store
uploaded_docs = {}


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if file.filename.endswith(".pdf"):
        file_bytes = await file.read()
        text = extract_text_from_pdf(file_bytes)
        uploaded_docs[file.filename] = text
        load_text_into_vectorstore(text)  # ← Load to FAISS
        return {"filename": file.filename, "status": "parsed", "chars": len(text)}


@app.get("/ping")
async def ping():
    return {"status": "alive"}


class Question(BaseModel):
    query: str


@app.post("/ask/")
async def ask(query: Question):
    answer = ask_question(query.query)
    return {"answer": answer}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
