#!/usr/bin/env python3
"""
RAG-MIDI Project Manager
Facilita o desenvolvimento e deployment do sistema RAG-MIDI
"""

import os
import sys
import subprocess
import argparse
import threading
from pathlib import Path

def run_backend():
    """Executa apenas o backend Streamlit"""
    print("🚀 Iniciando backend RAG-MIDI...")
    os.chdir("backend")
    subprocess.run([sys.executable, "run_rag_midi.py", "--run"])

def run_frontend():
    """Executa apenas o frontend Next.js"""
    print("🚀 Iniciando frontend Next.js...")
    os.chdir("frontend")
    subprocess.run(["npm", "run", "dev"])

def run_fullstack():
    """Executa backend e frontend simultaneamente"""
    print("🚀 Iniciando stack completo RAG-MIDI...")
    
    def run_backend_thread():
        os.chdir("backend")
        subprocess.run([sys.executable, "run_rag_midi.py", "--run"])
    
    def run_frontend_thread():
        os.chdir("frontend") 
        subprocess.run(["npm", "run", "dev"])
    
    # Criar threads para executar backend e frontend simultaneamente
    backend_thread = threading.Thread(target=run_backend_thread)
    frontend_thread = threading.Thread(target=run_frontend_thread)
    
    backend_thread.start()
    frontend_thread.start()
    
    try:
        backend_thread.join()
        frontend_thread.join()
    except KeyboardInterrupt:
        print("\n⏹️ Parando aplicação...")

def setup_project():
    """Configuração inicial do projeto"""
    print("⚙️ Configurando projeto RAG-MIDI...")
    
    # Backend setup
    print("📦 Instalando dependências do backend...")
    os.chdir("backend")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Frontend setup
    print("📦 Instalando dependências do frontend...")
    os.chdir("../frontend")
    subprocess.run(["npm", "install"])
    
    os.chdir("..")
    print("✅ Projeto configurado com sucesso!")

def build_index():
    """Reconstrói o índice FAISS"""
    print("🔄 Reconstruindo índice...")
    os.chdir("backend")
    subprocess.run([sys.executable, "run_rag_midi.py", "--rebuild"])

def test_system():
    """Testa o sistema RAG"""
    print("🧪 Testando sistema...")
    os.chdir("backend")
    subprocess.run([sys.executable, "run_rag_midi.py", "--test"])

def main():
    parser = argparse.ArgumentParser(description="RAG-MIDI Project Manager")
    parser.add_argument("--setup", action="store_true", help="Configurar projeto inicial")
    parser.add_argument("--backend", action="store_true", help="Executar apenas backend")
    parser.add_argument("--frontend", action="store_true", help="Executar apenas frontend")
    parser.add_argument("--fullstack", action="store_true", help="Executar stack completo")
    parser.add_argument("--build-index", action="store_true", help="Reconstruir índice FAISS")
    parser.add_argument("--test", action="store_true", help="Testar sistema")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        print("🎵 RAG-MIDI Project Manager")
        print("=" * 40)
        print("Use --help para ver opções disponíveis")
        print("\nComandos principais:")
        print("  --setup      Configurar projeto inicial")
        print("  --backend    Executar backend Streamlit")
        print("  --frontend   Executar frontend Next.js")
        print("  --fullstack  Executar stack completo")
        return
    
    if args.setup:
        setup_project()
    
    if args.backend:
        run_backend()
    
    if args.frontend:
        run_frontend()
    
    if args.fullstack:
        run_fullstack()
    
    if args.build_index:
        build_index()
    
    if args.test:
        test_system()

if __name__ == "__main__":
    main() 