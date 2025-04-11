# frontend/app.py
import pandas as pd
import pygame
import streamlit as st
from rag_system import MusicRAGSystem
import os

# Configuração da página
st.set_page_config(
    page_title="RAG-MIDI",
    page_icon="🎵",
    layout="wide"
)

# Título e descrição
st.title("🎵 RAG-MIDI")
st.subheader("Assistente de Composição Musical")

st.markdown("""
Este sistema usa Retrieval-Augmented Generation (RAG) para encontrar trechos musicais
que correspondem à sua descrição. Digite um prompt descrevendo o tipo de música
que você está procurando, e o sistema encontrará exemplos similares.
""")


# Inicializar sistema RAG
@st.cache_resource
def load_rag_system():
    return MusicRAGSystem("../data/embeddings")


rag = load_rag_system()


# Função para tocar MIDI
def play_midi(midi_path):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(midi_path)
        pygame.mixer.music.play()
        return True
    except Exception as e:
        st.error(f"Erro ao tocar MIDI: {e}")
        return False


# Consulta
query = st.text_input("Descreva o tipo de música que você está procurando:",
                      placeholder="Ex: Uma melodia de piano calma em Lá menor")

top_k = st.slider("Número de resultados", min_value=1, max_value=10, value=5)

if query:
    with st.spinner("Buscando trechos musicais..."):
        results = rag.get_similar_segments(query, top_k=top_k)

    if results:
        st.success(
            f"Encontrados {len(results)} trechos musicais relacionados!")

        for i, result in enumerate(results):
            with st.expander(f"Resultado {i+1}: {result['description'][:100]}...", expanded=i == 0):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(
                        f"**Descrição completa:** {result['description']}")
                    st.markdown(
                        f"**Similaridade:** {result['similarity']:.2%}")
                    st.markdown(
                        f"**Instrumentos:** {', '.join(result.get('instruments', ['Desconhecido']))}")
                    st.markdown(f"**Tempo:** {result.get('tempo', 0):.1f} BPM")
                    st.markdown(
                        f"**Duração:** {result.get('duration', 0):.2f} segundos")

                with col2:
                    st.markdown("**Controles:**")
                    play_button = st.button(f"▶️ Tocar", key=f"play_{i}")
                    stop_button = st.button(f"⏹️ Parar", key=f"stop_{i}")

                    if play_button:
                        if play_midi(result['path']):
                            st.session_state[f"playing_{i}"] = True

                    if stop_button and pygame.mixer.get_init():
                        pygame.mixer.music.stop()
                        st.session_state[f"playing_{i}"] = False

                # Exibir caminho para o arquivo
                st.markdown(f"**Arquivo:** `{result.get('path', 'N/A')}`")
    else:
        st.warning("Nenhum resultado encontrado. Tente uma descrição diferente.")

# Rodapé
st.markdown("---")
st.markdown(
    "Desenvolvido como projeto de pesquisa para Moises | Embedding & Tokenização Musical")

# Caminho absoluto
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
commu_meta_path = os.path.join(base_dir, 'data', 'commu_meta.csv')

# Use este caminho ao ler o CSV
df = pd.read_csv(commu_meta_path)
