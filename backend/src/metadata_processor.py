"""
Processador de metadados para unificar e preparar dados dos diferentes datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
from typing import Dict, List, Optional
from config import METADATA_FILES, METADATA_PICKLE, MIDI_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetadataProcessor:
    def __init__(self):
        self.unified_df = None
        self.datasets = {}
        
    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Carrega todos os datasets de metadados"""
        logger.info("Carregando datasets de metadados...")
        
        datasets = {}
        
        # ComMU
        if METADATA_FILES["commu"].exists():
            try:
                df = pd.read_csv(METADATA_FILES["commu"])
                df['dataset_source'] = 'commu'
                datasets['commu'] = df
                logger.info(f"ComMU carregado: {len(df)} registros")
            except Exception as e:
                logger.error(f"Erro ao carregar ComMU: {e}")
        
        # MidiCaps 
        if METADATA_FILES["midicaps"].exists():
            try:
                df = pd.read_csv(METADATA_FILES["midicaps"])
                df['dataset_source'] = 'midicaps'
                datasets['midicaps'] = df
                logger.info(f"MidiCaps carregado: {len(df)} registros")
            except Exception as e:
                logger.error(f"Erro ao carregar MidiCaps: {e}")
        
        # E-GMD
        if METADATA_FILES["egmd"].exists():
            try:
                df = pd.read_csv(METADATA_FILES["egmd"])
                df['dataset_source'] = 'egmd'
                datasets['egmd'] = df
                logger.info(f"E-GMD carregado: {len(df)} registros")
            except Exception as e:
                logger.error(f"Erro ao carregar E-GMD: {e}")
        
        self.datasets = datasets
        return datasets
    
    def standardize_columns(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Padroniza colunas para um formato unificado"""
        
        # Mapeamento de colunas comuns para formato padronizado
        column_mapping = {
            # Campos básicos
            'title': ['title', 'song_title', 'track_title', 'name'],
            'artist': ['artist', 'composer', 'author'],
            'genre': ['genre', 'style', 'music_style', 'category'],
            'tempo': ['tempo', 'bpm', 'avg_tempo'],
            'duration': ['duration', 'length', 'song_length'],
            'key': ['key', 'key_signature', 'tonic', 'audio_key'],
            'time_signature': ['time_signature', 'time_sig', 'meter'],
            
            # Campos técnicos
            'num_tracks': ['num_tracks', 'track_count', 'n_tracks'],
            'num_notes': ['num_notes', 'note_count', 'total_notes'],
            'instruments': ['instruments', 'instrument', 'programs', 'instrument_summary'],
            
            # Identificadores (específicos por dataset)
            'filename': ['filename', 'file_name', 'midi_filename', 'path', 'location'],
            'file_id': ['file_id', 'id', 'midi_id', 'uid']
        }
        
        standardized_df = df.copy()
        
        # Aplicar mapeamento de colunas
        for standard_col, possible_cols in column_mapping.items():
            for col in possible_cols:
                if col in df.columns and standard_col not in standardized_df.columns:
                    standardized_df[standard_col] = df[col]
                    break
        
        # Preservar colunas específicas importantes de cada dataset
        if dataset_name == 'commu':
            # Manter coluna 'id' para ComMU
            if 'id' in df.columns:
                standardized_df['id'] = df['id']
        elif dataset_name == 'midicaps':
            # Manter coluna 'location' para MidiCaps
            if 'location' in df.columns:
                standardized_df['location'] = df['location']
            # Usar caption como título (primeiras 100 caracteres)
            if 'caption' in df.columns:
                captions = df['caption'].astype(str)
                standardized_df['title'] = captions.str[:100] + '...'
        elif dataset_name == 'egmd':
            # Manter coluna 'midi_filename' para E-GMD
            if 'midi_filename' in df.columns:
                standardized_df['midi_filename'] = df['midi_filename']
            # Usar style como gênero
            if 'style' in df.columns:
                standardized_df['genre'] = df['style']
        
        # Adicionar colunas padrão se não existirem
        required_columns = [
            'title', 'artist', 'genre', 'tempo', 'duration', 
            'filename', 'dataset_source'
        ]
        
        for col in required_columns:
            if col not in standardized_df.columns:
                standardized_df[col] = 'Unknown'
        
        return standardized_df
    
    def create_searchable_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria campo de texto pesquisável combinando metadados"""
        
        def combine_metadata(row):
            parts = []
            
            # Adicionar campos principais
            for field in ['title', 'artist', 'genre']:
                if pd.notna(row.get(field)) and str(row.get(field)) != 'Unknown':
                    parts.append(str(row[field]))
            
            # Adicionar informações técnicas
            if pd.notna(row.get('tempo')):
                parts.append(f"tempo {row['tempo']}")
            
            if pd.notna(row.get('key')):
                parts.append(f"key {row['key']}")
            
            if pd.notna(row.get('time_signature')):
                parts.append(f"time signature {row['time_signature']}")
            
            # Adicionar instrumentos se disponível
            if pd.notna(row.get('instruments')):
                parts.append(f"instruments {row['instruments']}")
            
            return ' '.join(parts)
        
        df['searchable_text'] = df.apply(combine_metadata, axis=1)
        return df
    
    def verify_file_exists(self, df: pd.DataFrame) -> pd.DataFrame:
        """Verifica se os arquivos MIDI realmente existem"""
        logger.info("Verificando existência dos arquivos MIDI...")
        
        def check_file_exists(row):
            dataset = row['dataset_source']
            
            # Estratégias diferentes por dataset
            if dataset == 'commu':
                # ComMU: usar coluna 'id' + .mid
                file_id = row.get('id', row.get('file_id', ''))
                if not file_id:
                    return None
                
                filename = f"{file_id}.mid"
                possible_paths = [
                    MIDI_DIR / "commu_midi" / "train" / "raw" / filename,
                    MIDI_DIR / "commu_midi" / "val" / "raw" / filename,
                ]
                
            elif dataset == 'midicaps':
                # MidiCaps: usar coluna 'location' diretamente
                location = row.get('location', row.get('filename', ''))
                if not location:
                    return None
                
                possible_paths = [
                    MIDI_DIR / location,
                ]
                
            elif dataset == 'egmd':
                # E-GMD: usar coluna 'midi_filename'
                midi_filename = row.get('midi_filename', row.get('filename', ''))
                if not midi_filename:
                    return None
                
                possible_paths = [
                    MIDI_DIR / "e-gmd-v1.0.0" / midi_filename,
                ]
                
            else:
                # Fallback genérico
                filename = row.get('filename', row.get('file_name', ''))
                if not filename:
                    return None
                    
                possible_paths = [
                    MIDI_DIR / filename,
                    MIDI_DIR / dataset / filename,
                ]
            
            # Verificar se algum dos caminhos existe
            for path in possible_paths:
                if path.exists():
                    # Monta o caminho relativo ao diretório MIDI
                    rel_path = path.relative_to(MIDI_DIR)
                    return rel_path.as_posix()  # Só o caminho relativo
            
            return None
        
        df['file_path'] = df.apply(check_file_exists, axis=1)
        
        # Manter apenas registros com arquivos existentes
        before_count = len(df)
        df = df[df['file_path'].notna()].copy()
        after_count = len(df)
        
        logger.info(f"Arquivos verificados: {after_count}/{before_count} encontrados")
        
        return df
    
    def process_all(self) -> pd.DataFrame:
        """Processa todos os datasets e cria DataFrame unificado"""
        logger.info("Iniciando processamento completo dos metadados...")
        
        # Carregar datasets
        datasets = self.load_datasets()
        
        if not datasets:
            raise ValueError("Nenhum dataset foi carregado!")
        
        unified_dfs = []
        
        # Processar cada dataset
        for name, df in datasets.items():
            logger.info(f"Processando dataset: {name}")
            
            # Padronizar colunas
            df_std = self.standardize_columns(df, name)
            
            # Criar texto pesquisável
            df_std = self.create_searchable_text(df_std)
            
            unified_dfs.append(df_std)
        
        # Unificar todos os DataFrames
        self.unified_df = pd.concat(unified_dfs, ignore_index=True, sort=False)
        
        # Verificar existência dos arquivos
        self.unified_df = self.verify_file_exists(self.unified_df)
        
        logger.info(f"Dataset unificado criado: {len(self.unified_df)} registros")
        
        # Salvar resultado
        self.save_processed_data()
        
        return self.unified_df
    
    def save_processed_data(self):
        """Salva dados processados em pickle"""
        logger.info(f"Salvando dados processados em: {METADATA_PICKLE}")
        
        with open(METADATA_PICKLE, 'wb') as f:
            pickle.dump(self.unified_df, f)
    
    def load_processed_data(self) -> Optional[pd.DataFrame]:
        """Carrega dados processados do pickle"""
        if METADATA_PICKLE.exists():
            logger.info("Carregando dados processados do cache...")
            with open(METADATA_PICKLE, 'rb') as f:
                self.unified_df = pickle.load(f)
            return self.unified_df
        return None

if __name__ == "__main__":
    processor = MetadataProcessor()
    df = processor.process_all()
    print(f"✅ Processamento concluído: {len(df)} registros unificados") 