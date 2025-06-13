#!/usr/bin/env python3
"""
Script principal para executar o sistema RAG-MIDI
"""

import sys
import os
import argparse
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def install_dependencies():
    """Instala dependências necessárias"""
    import subprocess
    
    print("📦 Instalando dependências...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", 
                              str(Path(__file__).parent / "requirements.txt")])
        print("✅ Dependências instaladas com sucesso!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao instalar dependências: {e}")
        return False
    return True

def test_rag_engine():
    """Testa o motor RAG"""
    print("🧪 Testando motor RAG...")
    
    try:
        from src.rag_engine import initialize_rag_engine
        
        # Inicializar motor
        engine = initialize_rag_engine(force_rebuild=False)
        
        # Teste de busca
        results = engine.search("piano jazz", top_k=3)
        
        print(f"✅ Motor RAG funcionando! Encontrados {len(results)} resultados.")
        
        if results:
            print("\n🔍 Exemplo de resultados:")
            for i, result in enumerate(results[:3], 1):
                print(f"  {i}. {result['title']} - {result['artist']} (Score: {result['score']:.3f})")
        
        # Estatísticas
        stats = engine.get_stats()
        print(f"\n📊 Total de arquivos indexados: {stats.get('total_files', 0):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste do motor RAG: {e}")
        return False

def run_web_app():
    """Executa a aplicação web"""
    print("🚀 Iniciando aplicação web RAG-MIDI...")
    
    try:
        import subprocess
        import os
        
        # Executar Streamlit
        subprocess.run([
            "streamlit", "run", 
            str(Path(__file__).parent / "src" / "streamlit_app.py"),
            "--server.port", "8501",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
        
    except Exception as e:
        print(f"❌ Erro ao executar aplicação web: {e}")
        print("Certifique-se de que o Streamlit está instalado: pip install streamlit")

def rebuild_index():
    """Reconstrói o índice FAISS"""
    print("🔄 Reconstruindo índice FAISS...")
    
    try:
        from src.rag_engine import initialize_rag_engine
        
        engine = initialize_rag_engine(force_rebuild=True)
        stats = engine.get_stats()
        
        print(f"✅ Índice reconstruído com sucesso!")
        print(f"📊 Total de arquivos indexados: {stats.get('total_files', 0):,}")
        
    except Exception as e:
        print(f"❌ Erro ao reconstruir índice: {e}")

def main():
    parser = argparse.ArgumentParser(description="RAG-MIDI System")
    parser.add_argument("--install", action="store_true", help="Instalar dependências")
    parser.add_argument("--test", action="store_true", help="Testar motor RAG")
    parser.add_argument("--rebuild", action="store_true", help="Reconstruir índice FAISS")
    parser.add_argument("--run", action="store_true", help="Executar aplicação web")
    
    args = parser.parse_args()
    
    if not any([args.install, args.test, args.rebuild, args.run]):
        # Execução padrão: aplicação web
        print("🎵 RAG-MIDI: Search & Discover")
        print("=" * 40)
        
        # Verificar se dependencies estão instaladas
        try:
            import reflex
            import sentence_transformers
            import faiss
        except ImportError:
            print("❌ Dependências não encontradas. Instalando...")
            if not install_dependencies():
                return
        
        run_web_app()
        return
    
    if args.install:
        install_dependencies()
    
    if args.test:
        test_rag_engine()
    
    if args.rebuild:
        rebuild_index()
    
    if args.run:
        run_web_app()

if __name__ == "__main__":
    main() 