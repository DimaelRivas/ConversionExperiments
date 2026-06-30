# Human Review Web App

Local FastAPI app for reviewing PDF-to-Markdown conversions with the source PDF on the left and editable markdown on the right.

## Setup

```bash
cd review_app
pip install -r requirements.txt
```

## Running

```bash
# Run from inside review_app/
uvicorn main:app --reload --port 8000
```

Open: http://localhost:8000

## Usage

1. Select a file from the dropdown - PDF loads left, markdown loads right
2. Click inside any orange marker block - PDF jumps to that page automatically
3. Edit the table in the markdown to match the source PDF
4. Click `✓ Remove Warning` or press `Alt+R` - the review warning wrapper is removed, Docling content stays, and the file auto-saves
5. Use `Alt+N` / `Alt+P` to move between markers without the mouse
6. `Ctrl+S` saves manually at any time

## Expected Directory Layout

```text
project_root/
├── data/           ← source PDFs
├── output/         ← markdown files to review
└── review_app/     ← this app
```

## Override Data Directories

```bash
DATA_DIR=/custom/pdfs OUTPUT_DIR=/custom/md uvicorn main:app --port 8000
```
