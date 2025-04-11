# src/run_pipeline.py
import os
import argparse
from tokenize_midi import tokenize_dataset
from generate_descriptions import generate_descriptions
from create_embeddings import create_embeddings
import time
import pandas as pd


def run_pipeline(commu_meta_path, output_dir="./data", limit=None):
    """Executa a pipeline completa de processamento."""
    start_time = time.time()

    # Criar diretórios
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tokenized"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "embeddings"), exist_ok=True)

    # Passo 1: Carregar metadados
    print("\n==== Passo 1: Carregando Metadados do Commu MIDI ====")
    df = pd.read_csv(commu_meta_path)
    if limit:
        df = df.head(limit)
    print(f"Carregados {len(df)} registros de metadados")

    # Passo 2: Tokenizar MIDIs
    print("\n==== Passo 2: Tokenizando Arquivos MIDI ====")
    tokens_pkl = os.path.join(output_dir, "tokenized", "midi_tokens.pkl")
    midi_tokens = tokenize_dataset(
        commu_meta_path, os.path.join(output_dir, "tokenized"))

    # Passo 3: Gerar descrições
    print("\n==== Passo 3: Gerando Descrições Musicais ====")
    descriptions_csv = os.path.join(output_dir, "midi_descriptions.csv")
    descriptions_df = generate_descriptions(
        commu_meta_path, descriptions_csv)

    # Passo 4: Criar embeddings
    print("\n==== Passo 4: Criando Embeddings ====")
    embeddings, index = create_embeddings(
        descriptions_csv,
        tokens_pkl,
        os.path.join(output_dir, "embeddings")
    )

    end_time = time.time()
    print(
        f"\n==== Pipeline Concluída em {end_time - start_time:.2f} segundos ====")
    print(f"Arquivos MIDI processados: {len(df)}")
    print(f"Resultados salvos em {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Execute a pipeline completa do MaestrAI-RAG')
    parser.add_argument('--commu_meta', type=str, required=True,
                        help='Caminho para o arquivo commu_meta.csv')
    parser.add_argument('--output_dir', type=str,
                        default='./data', help='Diretório de saída')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limite de arquivos MIDI a processar')

    args = parser.parse_args()
    run_pipeline(args.commu_meta, args.output_dir, args.limit)
