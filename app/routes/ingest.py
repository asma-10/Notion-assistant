import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.ingestion import ingest_documents
from app.schemas import IngestResponse

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

""" Accepts file uploads, saves them to a temp directory, runs the ingestion pipeline, then cleans up. """

@router.post("/", response_model=IngestResponse)
async def ingest_files(files: list[UploadFile] = File(...)):
    """
    Upload one or more documents to be indexed.
    Supported formats: PDF, TXT, MD, DOCX.
    """
    saved_paths = []
    tmp_dir = tempfile.mkdtemp()

    try:
        for file in files:
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"File type '{ext}' not supported. Use: {ALLOWED_EXTENSIONS}",
                )

            dest = os.path.join(tmp_dir, file.filename)
            content = await file.read()
            with open(dest, "wb") as f:
                f.write(content)
            saved_paths.append(dest)

        result = ingest_documents(saved_paths)

        return IngestResponse(
            message="Ingestion complete",
            node_ingested=result["nodes_ingested"],
            files=result["files"],
        )

    finally:
        # Always clean up temp files, even if ingestion fails
        for path in saved_paths:
            if os.path.exists(path):
                os.remove(path)