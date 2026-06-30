import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("review_app")

APP_DIR = Path(__file__).parent.resolve()
STATIC_DIR = APP_DIR / "static"

DATA_DIR = Path(os.getenv("DATA_DIR", "../data")).expanduser().resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "../output")).expanduser().resolve()

MARKER_START = "<!-- ⚠️ REVIEW NEEDED"

app = FastAPI(title="PDF Markdown Review App")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def log_paths() -> None:
    logger.info("Review app DATA_DIR: %s", DATA_DIR)
    logger.info("Review app OUTPUT_DIR: %s", OUTPUT_DIR)


def validate_name(name: str) -> None:
    if not name or ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid file name")


def markdown_path(name: str) -> Path:
    validate_name(name)
    return OUTPUT_DIR / f"{name}.md"


def pdf_path(name: str) -> Path:
    validate_name(name)
    return DATA_DIR / f"{name}.pdf"


def count_markers(content: str) -> int:
    return content.count(MARKER_START)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
async def api_status() -> JSONResponse:
    return JSONResponse(
        {
            "data_dir": str(DATA_DIR),
            "output_dir": str(OUTPUT_DIR),
            "data_exists": DATA_DIR.exists() and DATA_DIR.is_dir(),
            "output_exists": OUTPUT_DIR.exists() and OUTPUT_DIR.is_dir(),
        }
    )


@app.get("/api/files")
async def api_files() -> JSONResponse:
    if not DATA_DIR.exists() or not DATA_DIR.is_dir():
        return JSONResponse([])
    if not OUTPUT_DIR.exists() or not OUTPUT_DIR.is_dir():
        return JSONResponse([])

    pdf_stems = {path.stem for path in DATA_DIR.glob("*.pdf") if path.is_file()}
    md_stems = {path.stem for path in OUTPUT_DIR.glob("*.md") if path.is_file()}

    files = []
    for name in sorted(pdf_stems & md_stems):
        md_path = OUTPUT_DIR / f"{name}.md"
        try:
            marker_count = count_markers(read_text(md_path))
        except OSError:
            marker_count = 0
        files.append(
            {
                "name": name,
                "pdf": f"{name}.pdf",
                "markdown": f"{name}.md",
                "marker_count": marker_count,
            }
        )
    return JSONResponse(files)


@app.get("/api/markdown/{name:path}")
async def get_markdown(name: str) -> PlainTextResponse:
    path = markdown_path(name)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Markdown file not found")
    return PlainTextResponse(read_text(path), media_type="text/plain; charset=utf-8")


@app.post("/api/markdown/{name:path}")
async def save_markdown(name: str, request: Request) -> JSONResponse:
    path = markdown_path(name)
    if not OUTPUT_DIR.exists() or not OUTPUT_DIR.is_dir():
        raise HTTPException(status_code=404, detail="Output directory not found")

    content = (await request.body()).decode("utf-8")
    path.write_text(content, encoding="utf-8")
    return JSONResponse({"status": "saved", "marker_count": count_markers(content)})


@app.get("/api/pdf/{name:path}")
async def get_pdf(name: str) -> FileResponse:
    path = pdf_path(name)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="PDF file not found")
    return FileResponse(
        path,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{path.name}"'},
    )
