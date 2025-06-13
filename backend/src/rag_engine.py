"""
Motor RAG para busca semântica em metadados MIDI
"""

import numpy as np
import pandas as pd
import faiss
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, FAISS_INDEX_FILE, TOP_K_RESULTS, SIMILARITY_THRESHOLD
from metadata_processor import MetadataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        self.model = None
        self.index = None
        self.metadata_df = None
        self.processor = MetadataProcessor()
        
    def load_embedding_model(self):
        """Carrega modelo de embeddings"""
        logger.info(f"Carregando modelo de embeddings: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        return self.model
    
    def prepare_data(self, force_rebuild: bool = False) -> pd.DataFrame:
        """Prepara e carrega dados de metadados"""
        
        if not force_rebuild:
            # Tentar carregar dados processados
            df = self.processor.load_processed_data()
            if df is not None:
                self.metadata_df = df
                return df
        
        # Processar dados do zero
        logger.info("Processando metadados...")
        df = self.processor.process_all()
        self.metadata_df = df
        return df
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Cria embeddings para lista de textos"""
        if self.model is None:
            self.load_embedding_model()
        
        logger.info(f"Criando embeddings para {len(texts)} textos...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        return embeddings
    
    def build_index(self, force_rebuild: bool = False):
        """Constrói índice FAISS para busca rápida"""
        
        index_exists = FAISS_INDEX_FILE.exists()
        
        if not force_rebuild and index_exists:
            logger.info("Carregando índice FAISS existente...")
            self.index = faiss.read_index(str(FAISS_INDEX_FILE))
            
            # Carregar dados processados também
            if self.metadata_df is None:
                self.prepare_data(force_rebuild=False)
            
            return
        
        logger.info("Construindo novo índice FAISS...")
        
        # Preparar dados
        df = self.prepare_data(force_rebuild)
        
        if df.empty:
            raise ValueError("Nenhum dado disponível para indexação!")
        
        # Criar embeddings
        texts = df['searchable_text'].fillna('').tolist()
        embeddings = self.create_embeddings(texts)
        
        # Construir índice FAISS
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
        
        # Normalizar embeddings para cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Adicionar ao índice
        self.index.add(embeddings.astype('float32'))
        
        # Salvar índice
        logger.info(f"Salvando índice FAISS: {FAISS_INDEX_FILE}")
        faiss.write_index(self.index, str(FAISS_INDEX_FILE))
        
        logger.info(f"✅ Índice construído: {self.index.ntotal} embeddings")
    
    def search(self, query: str, top_k: int = None, filters: Dict = None) -> List[Dict]:
        """
        Busca semântica por MIDIs
        
        Args:
            query: Texto de busca
            top_k: Número de resultados (padrão: TOP_K_RESULTS)
            filters: Filtros adicionais (genre, artist, tempo_range, etc.)
        
        Returns:
            Lista de resultados com metadados e scores
        """
        if top_k is None:
            top_k = TOP_K_RESULTS
            
        if self.index is None or self.metadata_df is None:
            raise ValueError("Índice não carregado! Execute build_index() primeiro.")
        
        logger.info(f"Buscando: '{query}'")
        
        # Para filtros específicos como time_signature, fazer busca híbrida
        if filters and 'time_signature' in filters and filters['time_signature']:
            return self._hybrid_search(query, top_k, filters)
        
        # Busca semântica normal
        # Se há filtros, buscar mais resultados para compensar a filtragem
        search_multiplier = 10 if filters else 2
        search_k = min(top_k * search_multiplier, self.index.ntotal)
        
        # Se a query está vazia e há filtros, fazer busca mais abrangente
        if not query.strip() and filters:
            query = "music midi song"  # Query genérica para capturar mais resultados
            search_k = min(1000, self.index.ntotal)  # Buscar mais resultados
        
        # Criar embedding da query
        query_embedding = self.create_embeddings([query])
        faiss.normalize_L2(query_embedding)
        
        # Buscar no índice
        scores, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        results = []
        
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Fim dos resultados válidos
                break
                
            if score < SIMILARITY_THRESHOLD:
                continue
                
            # Obter metadados
            metadata = self.metadata_df.iloc[idx].to_dict()
            
            # Aplicar filtros se especificados
            if filters and not self._apply_filters(metadata, filters):
                continue
            
            result = {
                'score': float(score),
                'metadata': metadata,
                'searchable_text': metadata.get('searchable_text', ''),
                'file_path': metadata.get('file_path'),
                'title': metadata.get('title', 'Unknown'),
                'artist': metadata.get('artist', 'Unknown'),
                'genre': metadata.get('genre', 'Unknown'),
                'dataset_source': metadata.get('dataset_source', 'Unknown')
            }
            
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        logger.info(f"Encontrados {len(results)} resultados")
        return results
    
    def _hybrid_search(self, query: str, top_k: int, filters: Dict) -> List[Dict]:
        """Busca híbrida que primeiro filtra por metadados e depois aplica similaridade semântica"""
        
        # Primeiro, filtrar o dataset por metadados
        filtered_df = self.metadata_df.copy()
        
        for filter_key, filter_value in filters.items():
            if not filter_value:
                continue
                
            if filter_key == 'time_signature':
                filtered_df = filtered_df[filtered_df['time_signature'] == filter_value]
            elif filter_key == 'genre':
                filtered_df = filtered_df[filtered_df['genre'].str.contains(filter_value, case=False, na=False)]
            elif filter_key == 'dataset':
                filtered_df = filtered_df[filtered_df['dataset_source'] == filter_value]
            elif filter_key == 'tempo_min':
                tempo_num = pd.to_numeric(filtered_df['tempo'], errors='coerce')
                filtered_df = filtered_df[tempo_num >= filter_value]
            elif filter_key == 'tempo_max':
                tempo_num = pd.to_numeric(filtered_df['tempo'], errors='coerce')
                filtered_df = filtered_df[tempo_num <= filter_value]
        
        if filtered_df.empty:
            logger.info("Nenhum resultado encontrado após filtragem")
            return []
        
        logger.info(f"Após filtragem: {len(filtered_df)} candidatos")
        
        # Se não há query de texto, retornar resultados filtrados ordenados aleatoriamente
        if not query.strip():
            sample_size = min(top_k, len(filtered_df))
            sampled = filtered_df.sample(n=sample_size) if len(filtered_df) > sample_size else filtered_df
            
            results = []
            for idx, row in sampled.iterrows():
                result = {
                    'score': 1.0,  # Score neutro para resultados filtrados
                    'metadata': row.to_dict(),
                    'searchable_text': row.get('searchable_text', ''),
                    'file_path': row.get('file_path'),
                    'title': row.get('title', 'Unknown'),
                    'artist': row.get('artist', 'Unknown'),
                    'genre': row.get('genre', 'Unknown'),
                    'dataset_source': row.get('dataset_source', 'Unknown')
                }
                results.append(result)
            
            return results
        
        # Com query de texto, aplicar busca semântica nos resultados filtrados
        query_embedding = self.create_embeddings([query])
        faiss.normalize_L2(query_embedding)
        
        # Calcular similaridade com cada resultado filtrado
        results_with_scores = []
        
        for idx, row in filtered_df.iterrows():
            # Obter o índice original no dataset completo para buscar o embedding
            original_idx = self.metadata_df.index.get_loc(idx)
            
            # Buscar o embedding mais próximo (método simples)
            if original_idx < self.index.ntotal:
                # Calcular score usando o embedding do índice
                embedding = self.index.reconstruct(original_idx)
                embedding = embedding.reshape(1, -1)
                score = np.dot(query_embedding, embedding.T)[0][0]
                
                result = {
                    'score': float(score),
                    'metadata': row.to_dict(),
                    'searchable_text': row.get('searchable_text', ''),
                    'file_path': row.get('file_path'),
                    'title': row.get('title', 'Unknown'),
                    'artist': row.get('artist', 'Unknown'),
                    'genre': row.get('genre', 'Unknown'),
                    'dataset_source': row.get('dataset_source', 'Unknown')
                }
                results_with_scores.append(result)
        
        # Ordenar por score e retornar top_k
        results_with_scores.sort(key=lambda x: x['score'], reverse=True)
        
        return results_with_scores[:top_k]
    
    def _apply_filters(self, metadata: Dict, filters: Dict) -> bool:
        """Aplica filtros aos resultados"""
        
        # Filtro por gênero
        if 'genre' in filters:
            metadata_genre = str(metadata.get('genre', '')).lower()
            filter_genre = str(filters['genre']).lower()
            if filter_genre not in metadata_genre:
                return False
        
        # Filtro por artista
        if 'artist' in filters:
            metadata_artist = str(metadata.get('artist', '')).lower()
            filter_artist = str(filters['artist']).lower()
            if filter_artist not in metadata_artist:
                return False
        
        # Filtro por dataset
        if 'dataset' in filters:
            if metadata.get('dataset_source') != filters['dataset']:
                return False
        
        # Filtro por time signature
        if 'time_signature' in filters and filters['time_signature']:
            metadata_time_sig = str(metadata.get('time_signature', '')).strip()
            filter_time_sig = str(filters['time_signature']).strip()
            if metadata_time_sig != filter_time_sig:
                return False
        
        # Filtro por tempo (BPM)
        if 'tempo_min' in filters or 'tempo_max' in filters:
            tempo = metadata.get('tempo')
            if tempo and str(tempo).replace('.', '').isdigit():
                tempo = float(tempo)
                if 'tempo_min' in filters and tempo < filters['tempo_min']:
                    return False
                if 'tempo_max' in filters and tempo > filters['tempo_max']:
                    return False
        
        return True
    
    def get_recommendations(self, file_path: str, top_k: int = 10) -> List[Dict]:
        """Obtém recomendações baseadas em um arquivo específico"""
        
        # Encontrar o arquivo no dataset
        matches = self.metadata_df[self.metadata_df['file_path'] == file_path]
        
        if matches.empty:
            return []
        
        # Usar o texto pesquisável do arquivo como query
        searchable_text = matches.iloc[0]['searchable_text']
        
        # Buscar similares
        results = self.search(searchable_text, top_k + 1)  # +1 para excluir o próprio arquivo
        
        # Remover o arquivo original dos resultados
        recommendations = [r for r in results if r['file_path'] != file_path]
        
        return recommendations[:top_k]
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas do dataset"""
        if self.metadata_df is None:
            return {}
        
        df = self.metadata_df
        
        stats = {
            'total_files': len(df),
            'datasets': df['dataset_source'].value_counts().to_dict(),
            'genres': df['genre'].value_counts().head(10).to_dict(),
            'artists': df['artist'].value_counts().head(10).to_dict(),
            'avg_tempo': df['tempo'].apply(pd.to_numeric, errors='coerce').mean(),
            'file_extensions': df['filename'].str.split('.').str[-1].value_counts().to_dict()
        }
        
        return stats

def initialize_rag_engine(force_rebuild: bool = False) -> RAGEngine:
    """Inicializa e retorna motor RAG pronto para uso"""
    engine = RAGEngine()
    engine.load_embedding_model()
    engine.build_index(force_rebuild=force_rebuild)
    return engine

if __name__ == "__main__":
    # Teste do motor RAG
    engine = initialize_rag_engine(force_rebuild=True)
    
    # Teste de busca
    results = engine.search("jazz piano upbeat", top_k=5)
    
    print(f"\n🔍 Resultados para 'jazz piano upbeat':")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} - {result['artist']} (Score: {result['score']:.3f})")
    
    # Estatísticas
    stats = engine.get_stats()
    print(f"\n📊 Estatísticas do dataset:")
    print(f"Total de arquivos: {stats['total_files']}")
    print(f"Datasets: {stats['datasets']}") 