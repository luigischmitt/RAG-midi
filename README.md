# RAG-midi

RAG-midi is a music recommendation and analysis system that leverages MIDI tokenization and similarity search to assist in musical composition and exploration.

## Features

- **MIDI Tokenization**: Convert MIDI files into token sequences for analysis.
- **Similarity Search**: Find similar musical pieces based on text queries.
- **Streamlit Interface**: Interactive web application for exploring and analyzing music.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/RAG-midi.git
   cd RAG-midi
   ```

2. **Set Up the Virtual Environment**:
   ```bash
   python3.9 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip setuptools
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit App**:
   ```bash
   streamlit run src/app.py
   ```

2. **Access the App**:
   Open your browser and go to `http://localhost:8501` to interact with the application.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.