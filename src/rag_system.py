# src/rag_system.py
import numpy as np
import pandas as pd
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
import pretty_midi
from miditok import REMI


class MusicRAGSystem:
    """Sistema RAG para assistência à composição musical."""

    def __init__(self, embeddings_dir="./data/embeddings", tokenizer_path=None):
        # Carregar índice FAISS
        self.index = faiss.read_index(os.path.join(
            embeddings_dir, "description_index.faiss"))

        # Carregar mapeamentos
        with open(os.path.join(embeddings_dir, "index_to_id.pkl"), "rb") as f:
            self.index_to_id = pickle.load(f)

        with open(os.path.join(embeddings_dir, "index_to_description.pkl"), "rb") as f:
            self.index_to_description = pickle.load(f)

        # Carregar modelo de embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Inicializar tokenizador se fornecido
        self.tokenizer = None
        if tokenizer_path:
            self.tokenizer = REMI.load_from_file(tokenizer_path)

    def get_midi_path(self, midi_id, split_data):
        """Retorna o caminho completo para um arquivo MIDI baseado no ID e split."""
        base_dir = os.path.join("data", "commu_midi", split_data, "raw")
        return os.path.join(base_dir, f"{midi_id}.mid")

    def search(self, query, top_k=5):
        """Busca trechos musicais baseados em uma consulta em texto."""
        # Converter consulta para embedding
        query_embedding = self.model.encode([query])

        # Normalizar para busca por cosseno
        faiss.normalize_L2(query_embedding)

        # Buscar exemplos similares
        distances, indices = self.index.search(query_embedding, top_k)

        # Montar resultados
        results = []
        for i, idx in enumerate(indices[0]):
            midi_id = self.index_to_id[int(idx)]
            results.append({
                'index': int(idx),
                'id': midi_id,
                'description': self.index_to_description[int(idx)],
                # Converter distância em similaridade
                'similarity': 1.0 - float(distances[0][i])
            })

        return results

    def get_midi_info(self, midi_id, split_data):
        """Obtém informações básicas de um arquivo MIDI."""
        try:
            midi_path = self.get_midi_path(midi_id, split_data)
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            return {
                'duration': midi_data.get_end_time(),
                'tempo': midi_data.get_tempo_changes()[1][0] if len(midi_data.get_tempo_changes()[1]) > 0 else 120,
                'instruments': [
                    pretty_midi.program_to_instrument_name(i.program)
                    for i in midi_data.instruments if not i.is_drum
                ],
                'has_drums': any(i.is_drum for i in midi_data.instruments)
            }
        except Exception as e:
            return {'error': str(e)}

    def get_similar_segments(self, query, top_k=5):
        """Busca e retorna informações completas sobre trechos similares."""
        results = self.search(query, top_k)

        # Ajuste o caminho para o arquivo CSV
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        commu_meta_path = os.path.join(base_dir, 'data', 'commu_meta.csv')
        df = pd.read_csv(commu_meta_path)
        df.set_index('id', inplace=True)

        # Adicionar informações detalhadas
        for result in results:
            midi_id = result['id']
            if midi_id in df.index:
                row = df.loc[midi_id]
                midi_info = self.get_midi_info(midi_id, row['split_data'])
                result.update(midi_info)
                result.update({
                    'genre': row['genre'],
                    'track_role': row['track_role'],
                    'instrument': row['inst'],
                    'key': row['audio_key'],
                    'bpm': row['bpm'],
                    'path': self.get_midi_path(midi_id, row['split_data'])
                })

        return results


# Exemplo de uso
if __name__ == "__main__":
    rag = MusicRAGSystem("./data/embeddings")
    results = rag.get_similar_segments("A jazz piano piece with walking bass")

    for i, result in enumerate(results):
        print(f"\nResultado {i+1} (Similaridade: {result['similarity']:.2f}):")
        print(f"Descrição: {result['description']}")
        print(f"ID: {result['id']}")
        print(f"Gênero: {result.get('genre', 'N/A')}")
        print(f"Instrumento: {result.get('instrument', 'N/A')}")
        print(f"Função: {result.get('track_role', 'N/A')}")
        print(f"Tonalidade: {result.get('key', 'N/A')}")
        print(f"Tempo: {result.get('bpm', 'N/A')} BPM")
