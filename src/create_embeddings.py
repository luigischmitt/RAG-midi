# src/create_embeddings.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from tqdm import tqdm


def create_embeddings(descriptions_csv, tokens_pkl, output_dir="./data/embeddings"):
    """Cria embeddings para os trechos MIDI e suas descrições."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Carregar dados
    df = pd.read_csv(descriptions_csv)
    with open(tokens_pkl, "rb") as f:
        midi_tokens = pickle.load(f)

    # Inicializar modelo para embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Modelo leve

    # Criar embeddings das descrições
    descriptions = df['description'].tolist()
    print("Gerando embeddings para descrições...")
    description_embeddings = model.encode(descriptions, show_progress_bar=True)

    # Criar índice FAISS para busca rápida
    print("Criando índice FAISS...")
    dimension = description_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    # Normalizar para busca por cosseno
    faiss.normalize_L2(description_embeddings)
    index.add(description_embeddings)

    # Salvar embeddings e índice
    with open(os.path.join(output_dir, "description_embeddings.npy"), "wb") as f:
        np.save(f, description_embeddings)

    faiss.write_index(index, os.path.join(
        output_dir, "description_index.faiss"))

    # Salvar mapeamento de índices para IDs dos arquivos
    index_to_id = {i: row['id'] for i, row in df.iterrows()}
    with open(os.path.join(output_dir, "index_to_id.pkl"), "wb") as f:
        pickle.dump(index_to_id, f)

    # Salvar mapeamento de índices para descrições
    index_to_description = {i: desc for i, desc in enumerate(descriptions)}
    with open(os.path.join(output_dir, "index_to_description.pkl"), "wb") as f:
        pickle.dump(index_to_description, f)

    print(f"Embeddings e índices salvos em {output_dir}")
    return description_embeddings, index


# Exemplo de uso
if __name__ == "__main__":
    embeddings, index = create_embeddings(
        "./data/midi_descriptions.csv",
        "./data/tokenized/midi_tokens.pkl"
    )
