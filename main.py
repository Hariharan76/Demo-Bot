from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
import groq

app = FastAPI()


groq.api_key = "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"

def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    reader = PdfReader(pdf_file.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chat_with_pdf_groq(query: str, pdf_text: str) -> str:
    response = groq.Completion.create(
        engine="gemma-7b-it", 
        prompt=f"PDF Content: {pdf_text}\n\nUser Query: {query}\nAI Response:",
        max_tokens=150
    )
    return response.choices[0].text.strip()

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type")
    pdf_text = extract_text_from_pdf(file)
    return JSONResponse(content={"pdf_text": pdf_text})

@app.post("/chat/")
async def chat_with_pdf(query: str, pdf_text: str):
    response = chat_with_pdf_groq(query, pdf_text)
    return JSONResponse(content={"response": response})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
