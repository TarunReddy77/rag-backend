from typing import List
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from utils.file_parser import extract_text_from_pdf

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
        return {"filename": file.filename, "status": "parsed", "chars": len(text)}
    else:
        return {"error": "Unsupported file type"}


@app.get("/ping")
async def ping():
    return {"status": "alive"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
