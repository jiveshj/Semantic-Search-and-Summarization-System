# api_service.py - FastAPI service for the semantic search system
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import tempfile
import shutil
from pathlib import Path
import asyncio
from datetime import datetime
import logging

# Import our semantic search system
from semantic_search_system import SemanticSearchSystem, SearchResult, Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Semantic Search & Summarization API",
    description="A scalable document search and summarization system using transformer models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global search system instance
search_system: Optional[SemanticSearchSystem] = None

# Pydantic models for API
class DocumentInput(BaseModel):
    title: str
    content: str
    source: str = "api"

class SearchQuery(BaseModel):
    query: str
    k: int = 10
    include_summary: bool = True

class SearchResultResponse(BaseModel):
    document_id: str
    title: str
    content: str
    source: str
    score: float
    summary: Optional[str]
    relevance_explanation: str
    metadata: Dict[str, Any]

class SystemStats(BaseModel):
    total_documents: int
    index_size: int
    embedding_dimension: int
    query_stats: Dict[str, Any]
    cache_size: int
    uptime: str

class HealthCheck(BaseModel):
    status: str
    timestamp: str
    system_ready: bool

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the search system on startup"""
    global search_system
    try:
        logger.info("Initializing Semantic Search System...")
        search_system = SemanticSearchSystem()
        
        # Load existing index if available
        index_path = os.getenv("INDEX_PATH", "./search_index")
        if os.path.exists(f"{index_path}.index"):
            search_system.load_system(index_path)
            logger.info(f"Loaded existing index with {len(search_system.vector_index.documents)} documents")
        else:
            logger.info("Starting with empty index")
            
        logger.info("Search system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize search system: {str(e)}")
        raise

# Health check endpoint
@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        system_ready=search_system is not None
    )

# Search endpoint
@app.post("/search", response_model=List[SearchResultResponse])
async def search_documents(query: SearchQuery):
    """Search for documents using semantic similarity"""
    if not search_system:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    try:
        results = search_system.search(
            query.query, 
            k=query.k, 
            include_summary=query.include_summary
        )
        
        # Convert to response format
        response = []
        for result in results:
            response.append(SearchResultResponse(
                document_id=result.document.id,
                title=result.document.title,
                content=result.document.content[:500] + "..." if len(result.document.content) > 500 else result.document.content,
                source=result.document.source,
                score=result.score,
                summary=result.summary,
                relevance_explanation=result.relevance_explanation,
                metadata=result.document.metadata
            ))
        
        return response
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Add document endpoint
@app.post("/documents")
async def add_document(document: DocumentInput):
    """Add a single document to the search index"""
    if not search_system:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    try:
        doc_id = search_system.add_document_from_text(
            document.title, 
            document.content,
            document.source
        )
        
        return {"document_id": doc_id, "status": "added", "message": "Document added successfully"}
        
    except Exception as e:
        logger.error(f"Add document error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add document: {str(e)}")

# Upload file endpoint
@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process a document file"""
    if not search_system:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    # Check file type
    allowed_extensions = {'.txt', '.pdf', '.docx', '.html', '.md'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        # Process file in background
        background_tasks.add_task(process_uploaded_file, tmp_path, file.filename)
        
        return {
            "filename": file.filename,
            "status": "processing",
            "message": "File uploaded and processing started"
        }
        
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

async def process_uploaded_file(file_path: str, original_filename: str):
    """Background task to process uploaded file"""
    try:
        # Process the file
        docs_added = search_system.ingest_documents([file_path])
        logger.info(f"Processed {original_filename}: {docs_added} documents added")
        
        # Clean up temporary file
        os.unlink(file_path)
        
    except Exception as e:
        logger.error(f"Error processing {original_filename}: {str(e)}")
        # Clean up on error
        if os.path.exists(file_path):
            os.unlink(file_path)

# Get document by ID
@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Get a specific document by ID"""
    if not search_system:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    document = search_system.get_document_by_id(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "document_id": document.id,
        "title": document.title,
        "content": document.content,
        "source": document.source,
        "metadata": document.metadata,
        "created_at": document.created_at.isoformat()
    }

# System statistics
@app.get("/stats", response_model=SystemStats)
async def get_system_stats():
    """Get system statistics"""
    if not search_system:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    stats = search_system.get_stats()
    
    return SystemStats(
        total_documents=stats['total_documents'],
        index_size=stats['index_size'],
        embedding_dimension=stats['embedding_dimension'],
        query_stats=stats['query_stats'],
        cache_size=stats['cache_size'],
        uptime="N/A"  # Could add actual uptime tracking
    )

# Save system state
@app.post("/admin/save")
async def save_system():
    """Save the current system state to disk"""
    if not search_system:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    try:
        index_path = os.getenv("INDEX_PATH", "./search_index")
        search_system.save_system(index_path)
        return {"status": "saved", "message": f"System state saved to {index_path}"}
        
    except Exception as e:
        logger.error(f"Save error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save system: {str(e)}")

# Batch processing endpoint
@app.post("/batch/process")
async def batch_process_directory(background_tasks: BackgroundTasks, directory_path: str):
    """Process all documents in a directory"""
    if not search_system:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    if not os.path.exists(directory_path):
        raise HTTPException(status_code=404, detail="Directory not found")
    
    # Find all supported files
    supported_extensions = {'.txt', '.pdf', '.docx', '.html', '.md'}
    file_paths = []
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if Path(file).suffix.lower() in supported_extensions:
                file_paths.append(os.path.join(root, file))
    
    if not file_paths:
        raise HTTPException(status_code=400, detail="No supported files found in directory")
    
    # Process in background
    background_tasks.add_task(process_directory_batch, file_paths, directory_path)
    
    return {
        "directory": directory_path,
        "files_found": len(file_paths),
        "status": "processing",
        "message": f"Started processing {len(file_paths)} files"
    }

async def process_directory_batch(file_paths: List[str], directory_path: str):
    """Background task to process directory of files"""
    try:
        docs_added = search_system.ingest_documents(file_paths)
        logger.info(f"Batch processed {directory_path}: {docs_added} documents added from {len(file_paths)} files")
        
        # Auto-save after batch processing
        index_path = os.getenv("INDEX_PATH", "./search_index")
        search_system.save_system(index_path)
        
    except Exception as e:
        logger.error(f"Batch processing error for {directory_path}: {str(e)}")

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api_service:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("ENVIRONMENT") == "development"
    )


