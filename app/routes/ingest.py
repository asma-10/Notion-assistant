import os
import tempfile
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from app.ingestion import ingest_documents
from app.schemas import IngestResponse
from app.ingestion import ingest_notion

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

        result = await run_in_threadpool(ingest_documents, saved_paths)

        return IngestResponse(
            message="Ingestion complete",
            node_ingested=result["nodes_ingested"],
            files=result["files"],
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

@router.post("/notion", response_model=IngestResponse)
async def ingest_from_notion():
    """Pull all configured Notion pages and index them."""
    result = await run_in_threadpool(ingest_notion)
    if result["nodes_ingested"] == 0:
        raise HTTPException(
            status_code=404,
            detail="No Notion content found. Check the token, the IDs, and that the pages are shared with the integration.",
        )
    return IngestResponse(
        message="Notion sync complete",
        node_ingested=result["nodes_ingested"],
        files=result["files"],
    )