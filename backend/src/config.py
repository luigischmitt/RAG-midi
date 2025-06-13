"""
Configurações para o RAG-MIDI System
"""

import os
from pathlib import Path

# Caminhos base
BASE_DIR = Path(__file__).parent.parent.parent  # Vai para a raiz do projeto
DATA_DIR = BASE_DIR / "data"
MIDI_DIR = DATA_DIR / "midi"
METADATA_DIR = DATA_DIR / "metadata"
INDEX_DIR = DATA_DIR / "index"

# Arquivos de metadados
METADATA_FILES = {
    "commu": METADATA_DIR / "commu_meta.csv",
    "midicaps": METADATA_DIR / "midicaps_meta.csv", 
    "egmd": METADATA_DIR / "e-gmd-v1.0.0.csv"
}

# Configurações do RAG
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Modelo rápido e eficiente
FAISS_INDEX_FILE = INDEX_DIR / "midi_embeddings.faiss"
METADATA_PICKLE = INDEX_DIR / "metadata_processed.pkl"

# Configurações da busca
TOP_K_RESULTS = 20
SIMILARITY_THRESHOLD = 0.3

# Configurações da interface
APP_TITLE = "🎵 RAG-MIDI: Search & Discover"
APP_DESCRIPTION = "Find the perfect MIDI file using natural language search"

# Configurações de Deploy e Bucket
USE_BUCKET_URLS = os.getenv("USE_BUCKET_URLS", "false").lower() == "true"
BUCKET_BASE_URL = os.getenv("BUCKET_BASE_URL", "https://storage.googleapis.com/rag_midi")
BUCKET_NAME = os.getenv("BUCKET_NAME", "rag_midi")

# Para Moises/Music.ai - ajustar conforme necessário
MOISES_BUCKET_URL = os.getenv("MOISES_BUCKET_URL", "https://storage.googleapis.com/rag_midi")

# Criar diretórios se não existirem
for dir_path in [DATA_DIR, MIDI_DIR, METADATA_DIR, INDEX_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True) 