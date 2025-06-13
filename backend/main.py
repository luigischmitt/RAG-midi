from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import sys
import json
import math
import io
from pathlib import Path

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from src.rag_engine import initialize_rag_engine

# Google Cloud Storage
from google.cloud import storage
import os

app = FastAPI(title="RAG-MIDI API")

# Permite CORS para o frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa o engine globalmente
engine = initialize_rag_engine(force_rebuild=False)

# Configurar cliente do GCS
bucket_name = "rag_midi"

# Inicializar cliente do GCS
try:
    # Tenta usar credenciais padrão (service account da VM ou ambiente local)
    gcs_client = storage.Client()
    gcs_bucket = gcs_client.bucket(bucket_name)
    print(f"✅ Google Cloud Storage client inicializado para bucket: {bucket_name}")
except Exception as e:
    print(f"⚠️ Erro ao inicializar GCS client: {e}")
    gcs_client = None
    gcs_bucket = None

def generate_signed_url(file_path: str) -> str:
    """Gera URL do endpoint de download local"""
    if not file_path:
        return ""
    
    # Retorna URL do nosso endpoint de proxy ao invés do bucket direto
    return f"http://localhost:8000/download/{file_path}"

def clean_float_values(obj):
    """Remove valores NaN e Infinity que causam erro no JSON"""
    if isinstance(obj, dict):
        return {k: clean_float_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_float_values(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    return obj

class SearchRequest(BaseModel):
    query: str
    genre: Optional[str] = None
    dataset: Optional[str] = None
    time_signature: Optional[str] = None
    tempo_min: Optional[float] = None
    tempo_max: Optional[float] = None
    top_k: Optional[int] = 10

@app.post("/search")
def search(req: SearchRequest):
    filters = {}
    if req.genre:
        filters["genre"] = req.genre
    if req.dataset:
        filters["dataset"] = req.dataset
    if req.time_signature:
        filters["time_signature"] = req.time_signature
    if req.tempo_min is not None:
        filters["tempo_min"] = req.tempo_min
    if req.tempo_max is not None:
        filters["tempo_max"] = req.tempo_max
    
    results = engine.search(req.query, top_k=req.top_k, filters=filters)
    # Adapta para o formato esperado pelo frontend
    formatted = []
    for r in results:
        m = r["metadata"]
        # Gerar signed URL para o arquivo
        signed_url = generate_signed_url(m.get("file_path", ""))
        
        # Limpar valores que podem ser NaN
        tempo = m.get("tempo", 0)
        if isinstance(tempo, float) and (math.isnan(tempo) or math.isinf(tempo)):
            tempo = 0
        
        score = r.get("score", 0)
        if isinstance(score, float) and (math.isnan(score) or math.isinf(score)):
            score = 0.0
        
        formatted.append({
            "id": str(m.get("id", m.get("file_path", ""))),
            "title": str(m.get("title", "Unknown")),
            "artist": str(m.get("artist", "Unknown")),
            "genre": str(m.get("genre", "Unknown")),
            "dataset": str(m.get("dataset_source", "Unknown")),
            "tempo": float(tempo),
            "time_signature": str(m.get("time_signature", "")),
            "description": str(m.get("description", "")),
            "file_path": signed_url,  # URL assinada temporária
            "score": float(score)
        })
    
    # Limpar qualquer valor problemático antes de retornar
    response = {"results": formatted}
    response = clean_float_values(response)
    
    return response

@app.get("/stats")
def stats():
    stats = engine.get_stats()
    # Limpar valores problemáticos
    stats = clean_float_values(stats)
    return stats

@app.get("/download/{file_path:path}")
async def download_file(file_path: str):
    """
    Endpoint proxy que baixa arquivos do GCS e serve para o frontend
    """
    if not gcs_client or not gcs_bucket:
        raise HTTPException(status_code=503, detail="Google Cloud Storage não disponível")
    
    try:
        # Obter o blob (arquivo) do bucket
        blob = gcs_bucket.blob(file_path)
        
        # Verificar se o arquivo existe
        if not blob.exists():
            raise HTTPException(status_code=404, detail="Arquivo não encontrado")
        
        # Baixar o conteúdo do arquivo
        file_content = blob.download_as_bytes()
        
        # Determinar o tipo de conteúdo
        content_type = "audio/midi"
        if file_path.endswith('.mid') or file_path.endswith('.midi'):
            content_type = "audio/midi"
        
        # Extrair nome do arquivo para o cabeçalho
        filename = file_path.split('/')[-1]
        
        # Retornar como stream para download
        return StreamingResponse(
            io.BytesIO(file_content),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(file_content))
            }
        )
        
    except Exception as e:
        print(f"Erro ao baixar arquivo {file_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 
    