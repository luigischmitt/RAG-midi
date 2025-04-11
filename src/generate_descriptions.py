# src/generate_descriptions.py
import pandas as pd
import numpy as np
from transformers import pipeline
import pickle
import os
from tqdm import tqdm


def create_musical_feature_string(row):
    """Converte características musicais em uma string para o modelo."""
    features = f"Key: {row['audio_key']}, "
    features += f"Tempo: {int(row['bpm'])} BPM, "
    features += f"Time signature: {row['time_signature']}, "
    features += f"Genre: {row['genre']}, "
    features += f"Track role: {row['track_role']}, "
    features += f"Instrument: {row['inst']}, "
    features += f"Pitch range: {row['pitch_range']}, "
    features += f"Number of measures: {row['num_measures']}, "
    features += f"Velocity range: {row['min_velocity']}-{row['max_velocity']}"

    return features


def generate_descriptions(commu_meta_path, output_file="./data/midi_descriptions.csv"):
    """Gera descrições musicais baseadas nas características extraídas."""
    df = pd.read_csv(commu_meta_path)

    # Criar strings de características
    df['feature_string'] = df.apply(create_musical_feature_string, axis=1)

    # Inicializar pipeline de geração (modelo pequeno que roda na CPU)
    try:
        generator = pipeline("text-generation", model="distilgpt2", device=-1)

        # Gerar descrições
        descriptions = []
        for feature_str in tqdm(df['feature_string']):
            prompt = f"A musical piece with {feature_str}. This composition is"
            result = generator(
                prompt,
                max_length=100,        # Aumentado de 50 para 100
                max_new_tokens=50,     # Adicionado para controlar novos tokens
                do_sample=True,
                temperature=0.7,
                truncation=True        # Adicionado para evitar o aviso
            )
            description = result[0]['generated_text'].replace(
                prompt, "").strip()
            descriptions.append(description)

        df['description'] = descriptions
    except Exception as e:
        print(f"Erro ao gerar descrições com LLM: {e}")
        print("Usando descrições básicas em vez disso...")

        # Fallback: Descrições baseadas em regras
        df['description'] = df.apply(lambda row:
                                     f"A {row['genre']} piece in {row['audio_key']} played by {row['inst']} with {row['track_role']} role.", axis=1)

    # Salvar CSV com descrições
    df.to_csv(output_file, index=False)
    print(f"Descrições salvas em {output_file}")
    return df


# Exemplo de uso
if __name__ == "__main__":
    descriptions_df = generate_descriptions("./data/commu_meta.csv")
