# src/tokenize_midi.py
from miditok import REMI, TokenizerConfig
import os
import pandas as pd
import pickle
from tqdm import tqdm
import symusic


def setup_tokenizer():
    """Configura o tokenizador REMI."""
    config = TokenizerConfig(
        # Configuração básica
        pitch_range=(21, 109),
        beat_res={(0, 4): 8, (4, 12): 4},
        num_velocities=32,

        # Tokens especiais
        special_tokens=['PAD', 'BOS', 'EOS', 'MASK'],

        # Configurações de tokenização
        use_chords=True,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=True,
        use_sustain_pedals=True,
        use_programs=True,

        # Configurações de tempo e compasso
        tempo_range=(40, 250),
        num_tempos=32,
        time_signature_range={
            4: [1, 2, 3, 4, 5, 6],
            8: [3, 6, 12]
        },

        # Configurações de programa/instrumento
        # -1 para drums, 0-127 para instrumentos
        programs=list(range(-1, 128)),
        one_token_stream_for_programs=True,

        # Configurações adicionais
        encode_ids_split='bar',
        use_pitchdrum_tokens=True
    )

    # Inicializar tokenizador com a configuração
    tokenizer = REMI(config)
    return tokenizer


def tokenize_dataset(commu_meta_path, output_dir="./data/tokenized"):
    """Tokeniza todos os MIDIs do dataset."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Inicializar tokenizador
    tokenizer = setup_tokenizer()

    # Carregar metadados
    df = pd.read_csv(commu_meta_path)

    # Dicionário para armazenar tokens
    midi_tokens = {}
    failed_files = []

    # Processar cada arquivo
    for i, row in tqdm(df.iterrows(), total=len(df)):
        midi_id = row['id']
        split_data = row['split_data']
        midi_path = os.path.join(
            "data/commu_midi", split_data, "raw", f"{midi_id}.mid")

        try:
            # Carregar arquivo MIDI usando symusic
            midi_score = symusic.Score(midi_path)
            # Tokenizar o arquivo usando o método correto
            tokens = tokenizer(midi_score)

            # Salvar os tokens
            midi_tokens[midi_id] = tokens
        except Exception as e:
            print(f"Erro ao tokenizar {midi_path}: {e}")
            failed_files.append(midi_path)

    # Salvar tokens e lista de falhas
    with open(os.path.join(output_dir, "midi_tokens.pkl"), "wb") as f:
        pickle.dump(midi_tokens, f)

    with open(os.path.join(output_dir, "failed_tokenization.txt"), "w") as f:
        for path in failed_files:
            f.write(f"{path}\n")

    print(
        f"Tokenização completa. {len(midi_tokens)} arquivos tokenizados com sucesso.")
    print(f"{len(failed_files)} arquivos falharam na tokenização.")

    return midi_tokens


# Exemplo de uso
if __name__ == "__main__":
    tokens = tokenize_dataset("./data/commu_meta.csv", "./data/tokenized")
