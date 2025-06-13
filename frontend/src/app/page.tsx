'use client';

import { useState, useEffect } from 'react';
import Image from 'next/image';

interface SearchResult {
  id: string;
  title: string;
  artist: string;
  genre: string;
  dataset: string;
  tempo: number;
  time_signature: string;
  description: string;
  file_path: string;
  score: number;
}

interface Statistics {
  total_files: number;
  genres: { [key: string]: number };
  datasets: { [key: string]: number };
  avg_tempo: number;
  time_signatures: { [key: string]: number };
}

export default function RAGMidiPage() {
  const [activeTab, setActiveTab] = useState<'search' | 'stats'>('search');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState({
    genre: '',
    dataset: '',
    timeSignature: '',
    tempoMin: '',
    tempoMax: ''
  });
  const [statistics, setStatistics] = useState<Statistics>({
    total_files: 225066,
    genres: {
      'Jazz': 45000,
      'Classical': 38000,
      'Electronic': 32000,
      'Rock': 28000,
      'Pop': 25000,
      'Blues': 18000,
      'Folk': 15000,
      'Hip Hop': 12000,
      'R&B': 12066
    },
    datasets: {
      'ComMU': 98000,
      'MidiCaps': 78000,
      'E-GMD': 49066
    },
    avg_tempo: 118,
    time_signatures: {
      '4/4': 180000,
      '3/4': 25000,
      '2/4': 12000,
      '6/8': 6000,
      '9/8': 1566,
      '12/8': 500,
      '5/4': 800,
      '7/8': 450,
      '2/2': 280,
      '3/8': 220,
      '5/8': 180,
      '7/4': 120,
      '11/8': 85,
      '15/8': 35
    }
  });

  const genres = ['', 'Jazz', 'Classical', 'Electronic', 'Rock', 'Pop', 'Blues', 'Folk', 'Hip Hop', 'R&B'];
  const datasets = ['', 'ComMU', 'MidiCaps', 'E-GMD'];
  const timeSignatures = ['', '4/4', '3/4', '2/4', '6/8', '9/8', '12/8', '5/4', '7/8', '2/2', '3/8', '5/8', '7/4', '11/8', '15/8'];

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          genre: filters.genre,
          dataset: filters.dataset,
          time_signature: filters.timeSignature,
          tempo_min: filters.tempoMin ? Number(filters.tempoMin) : undefined,
          tempo_max: filters.tempoMax ? Number(filters.tempoMax) : undefined,
          top_k: 10
        }),
      });
      const data = await response.json();
      setResults(data.results);
    } catch (error) {
      console.error('Search error:', error);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = (result: SearchResult) => {
    window.open(result.file_path, '_blank');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const renderSearchTab = () => (
    <div className="tab-content">
      <div className="search-container">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Search for MIDI files... (e.g., 'jazz piano slow', 'classical violin')"
          className="search-input"
        />
        
        <div className="button-group">
          <button
            onClick={handleSearch}
            disabled={loading || !query.trim()}
            className="search-button"
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
          
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="filters-toggle"
          >
            {showFilters ? (
              <>
                <Image src="/down_arrow.png" alt="Hide Filters" width={15} height={8} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }} />
                Hide Filters
              </>
            ) : (
              'Show Filters'
            )}
          </button>
        </div>
      </div>

      <div className={`filters ${showFilters ? '' : 'hidden'}`}>
        <div className="filter-group">
          <label className="filter-label">Genre</label>
          <select
            value={filters.genre}
            onChange={(e) => setFilters({ ...filters, genre: e.target.value })}
            className="filter-select"
          >
            {genres.map(genre => (
              <option key={genre} value={genre}>
                {genre || 'All Genres'}
              </option>
            ))}
          </select>
        </div>

        <div className="filter-group">
          <label className="filter-label">Dataset</label>
          <select
            value={filters.dataset}
            onChange={(e) => setFilters({ ...filters, dataset: e.target.value })}
            className="filter-select"
          >
            {datasets.map(dataset => (
              <option key={dataset} value={dataset}>
                {dataset || 'All Datasets'}
              </option>
            ))}
          </select>
        </div>

        <div className="filter-group">
          <label className="filter-label">Time Signature</label>
          <select
            value={filters.timeSignature}
            onChange={(e) => setFilters({ ...filters, timeSignature: e.target.value })}
            className="filter-select"
          >
            {timeSignatures.map(ts => (
              <option key={ts} value={ts}>
                {ts || 'All Time Signatures'}
              </option>
            ))}
          </select>
        </div>

        <div className="filter-group">
          <label className="filter-label">Tempo (BPM)</label>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <input
              type="number"
              placeholder="Min"
              value={filters.tempoMin}
              onChange={(e) => setFilters({ ...filters, tempoMin: e.target.value })}
              className="filter-input"
              style={{ width: '50%' }}
            />
            <input
              type="number"
              placeholder="Max"
              value={filters.tempoMax}
              onChange={(e) => setFilters({ ...filters, tempoMax: e.target.value })}
              className="filter-input"
              style={{ width: '50%' }}
            />
          </div>
        </div>
      </div>

      <div className="results">
        {loading && (
          <div className="loading">
            <p>🎵 Searching through {statistics.total_files.toLocaleString()} MIDI files...</p>
          </div>
        )}

        {!loading && results.length > 0 && (
          <>
            <div className="results-header">
              Found {results.length} results
            </div>
            {results.map(result => (
              <div key={result.id} className="result-item">
                <div className="result-title">{result.title}</div>
                <div className="result-artist">by {result.artist}</div>
                <div className="result-meta">
                  <span>🎼 {result.genre}</span>
                  <span>📁 {result.dataset}</span>
                  <span>🥁 {result.tempo} BPM</span>
                  <span>🎵 {result.time_signature}</span>
                  <span>⭐ {(result.score * 100).toFixed(1)}%</span>
                </div>
                <div className="result-description">{result.description}</div>
                <button
                  onClick={() => handleDownload(result)}
                  className="download-button"
                >
                  ⬇️ Download MIDI
                </button>
              </div>
            ))}
          </>
        )}

        {!loading && query && results.length === 0 && (
          <div className="loading">
            <p>No results found. Try a different search term or adjust filters.</p>
          </div>
        )}
      </div>
    </div>
  );

  const renderStatsTab = () => (
    <div className="tab-content">
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-number">{statistics.total_files.toLocaleString()}</div>
          <div className="stat-label">Total MIDI Files</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">{Object.keys(statistics.genres).length}</div>
          <div className="stat-label">Music Genres</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">{Object.keys(statistics.datasets).length}</div>
          <div className="stat-label">Datasets</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">{statistics.avg_tempo}</div>
          <div className="stat-label">Average BPM</div>
        </div>
      </div>

      <div className="chart-container">
        <div className="chart-title">🎼 Files by Genre</div>
        <div className="stats-grid">
          {Object.entries(statistics.genres).map(([genre, count]) => (
            <div key={genre} className="stat-card">
              <div className="stat-number">{count.toLocaleString()}</div>
              <div className="stat-label">{genre}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="chart-container">
        <div className="chart-title">📁 Files by Dataset</div>
        <div className="stats-grid">
          {Object.entries(statistics.datasets).map(([dataset, count]) => (
            <div key={dataset} className="stat-card">
              <div className="stat-number">{count.toLocaleString()}</div>
              <div className="stat-label">{dataset}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="chart-container">
        <div className="chart-title">🎵 Files by Time Signature</div>
        <div className="stats-grid">
          {Object.entries(statistics.time_signatures).map(([ts, count]) => (
            <div key={ts} className="stat-card">
              <div className="stat-number">{count.toLocaleString()}</div>
              <div className="stat-label">{ts}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  return (
    <div className="container">
      <div className="main-content">
        <div className="header">
          <div className="logo">
            <Image
              src="/moises_full_logo.png"
              alt="Moises Logo"
              width={280}
              height={160}
              priority
            />
          </div>
          <h1 className="title">RAG MIDI</h1>
          <p className="subtitle">
            AI-powered MIDI search engine - Find the perfect MIDI file using natural language
          </p>
        </div>

        <div className="tabs">
          <button
            className={`tab ${activeTab === 'search' ? 'active' : ''}`}
            onClick={() => setActiveTab('search')}
          >
            🔍 Search
          </button>
          <button
            className={`tab ${activeTab === 'stats' ? 'active' : ''}`}
            onClick={() => setActiveTab('stats')}
          >
            📊 Statistics
          </button>
        </div>

        {activeTab === 'search' ? renderSearchTab() : renderStatsTab()}
      </div>

      <div className="footer">
        <p>Powered by Moises AI • RAG-MIDI Search Engine</p>
        <p>Built with Next.js, React, and advanced semantic search</p>
      </div>
    </div>
  );
}
