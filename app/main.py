from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.inference import generator
from app.schemas import GenerationRequest, GenerationResponse


ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(title="Urdu Question Generation")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/api/generate", response_model=GenerationResponse)
def generate_questions(request: GenerationRequest):
    text = " ".join(request.text.split())
    if text.count("<ans>") != 1 or text.count("</ans>") != 1:
        raise HTTPException(status_code=400, detail="Add exactly one <ans>...</ans> span.")
    if text.index("<ans>") >= text.index("</ans>"):
        raise HTTPException(status_code=400, detail="The answer span is invalid.")

    try:
        return generator.generate(text)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error