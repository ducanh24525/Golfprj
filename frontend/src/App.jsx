import React, { useState, useRef } from 'react';
import { UploadCloud, Play, Activity, CheckCircle2, AlertCircle } from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import './index.css';

function App() {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragging(true);
    } else if (e.type === 'dragleave') {
      setIsDragging(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (selectedFile) => {
    if (!selectedFile.type.startsWith('video/')) {
      setError('Please upload a valid video file (mp4, avi, etc).');
      return;
    }
    setFile(selectedFile);
    setError('');
    setResult(null);
  };

  const handleSubmit = async () => {
    if (!file) return;

    setLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Assuming FastAPI is running on localhost:8000
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const data = await response.json();

      // Transform data for Recharts
      // data.probabilities is an array of length T, each item is array of 9 probs
      const chartData = data.probabilities.map((probs, frameIdx) => {
        const frameData = { frame: frameIdx };
        data.event_names.forEach((name, i) => {
          frameData[name] = probs[i];
        });
        return frameData;
      });

      setResult({
        videoUrl: `http://localhost:8000${data.video_url}`,
        chartData,
        eventNames: data.event_names,
        eventSegments: data.event_segments || []
      });
    } catch (err) {
      setError(err.message || 'An error occurred during processing.');
    } finally {
      setLoading(false);
    }
  };

  const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b'];

  return (
    <div className="app-container">
      <header>
        <h1>Golf Pose</h1>
        <p>Advanced Swing Event Detection</p>
      </header>

      {!result && (
        <div className="glass-panel">
          <div className="section-title">
            <UploadCloud /> Upload Swing Video
          </div>

          <div
            className={`upload-area ${isDragging ? 'drag-active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              onChange={handleChange}
            />
            {file ? (
              <>
                <CheckCircle2 size={48} className="upload-icon" style={{ color: 'var(--success)' }} />
                <div className="upload-text">{file.name}</div>
                <div className="upload-subtext">Click or drag to replace</div>
              </>
            ) : (
              <>
                <UploadCloud size={48} className="upload-icon" />
                <div className="upload-text">Drag & Drop your video here</div>
                <div className="upload-subtext">or click to browse files</div>
              </>
            )}
          </div>

          {error && (
            <div style={{ color: 'var(--danger)', marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <AlertCircle size={20} /> {error}
            </div>
          )}

          {loading ? (
            <div className="loading-container">
              <div className="spinner"></div>
              <p>Analyzing swing... This may take a moment.</p>
            </div>
          ) : (
            <div style={{ textAlign: 'center' }}>
              <button
                className="btn-primary"
                onClick={(e) => { e.stopPropagation(); handleSubmit(); }}
                disabled={!file}
              >
                <Activity size={20} /> Analyze Swing
              </button>
            </div>
          )}
        </div>
      )}

      {result && !loading && (
        <div className="results-grid">
          <div className="glass-panel">
            <div className="section-title">
              <Play /> Annotated Video
            </div>
            <div className="video-container">
              <video src={result.videoUrl} controls autoPlay loop />
            </div>
          </div>

          <div className="glass-panel">
            <div className="section-title">
              <Activity /> Swing Event Timeline
            </div>

            {result.eventSegments && result.eventSegments.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', marginTop: '1rem' }}>
                {result.eventSegments.map((seg, idx) => {
                  const duration = seg.end_time_sec - seg.start_time_sec;
                  return (
                    <div
                      key={idx}
                      className="glass-panel"
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        padding: '1.25rem',
                        borderLeft: `5px solid ${colors[idx % colors.length]}`,
                        background: 'rgba(255, 255, 255, 0.03)',
                        animation: 'fadeIn 0.5s ease-out'
                      }}
                    >
                      <div>
                        <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 600, color: '#fff' }}>
                          {seg.name}
                        </h3>
                        <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                          Khung hình: {seg.start_frame} - {seg.end_frame}
                        </p>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <div style={{ fontSize: '1.1rem', fontWeight: 700, color: colors[idx % colors.length] }}>
                          {seg.start_time_sec.toFixed(2)}s - {seg.end_time_sec.toFixed(2)}s
                        </div>
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.1rem' }}>
                          Thời lượng: {duration.toFixed(2)}s
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p style={{ color: 'var(--text-secondary)', textAlign: 'center', padding: '2rem' }}>
                Không tìm thấy sự kiện nào có độ tin cậy cao (&gt;50%).
              </p>
            )}
          </div>
        </div>
      )}

      {result && (
        <div style={{ textAlign: 'center', marginTop: '2rem' }}>
          <button className="btn-primary" onClick={() => { setResult(null); setFile(null); }}>
            <UploadCloud size={20} /> Analyze Another Video
          </button>
        </div>
      )}
    </div>
  );
}

export default App;
