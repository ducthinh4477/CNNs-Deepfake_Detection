'use client';

import { useState, useRef } from 'react';
import axios from 'axios';
import {
  Upload,
  Shield,
  ShieldAlert,
  ShieldCheck,
  Activity,
  Eye,
  Waves,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Loader2,
  Image as ImageIcon,
  Info,
  Zap
} from 'lucide-react';

// API Base URL: Uses environment variable in production, localhost in development
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://deepscan-api.onrender.com/';

// Model info constant (synced with backend)
const MODEL_INFO = {
  name: 'Custom CNN',
  dataset: 'CIFAKE',
  accuracy: 92.0,
  version: '1.0',
};

// Gauge Chart Component
function GaugeChart({ value, isReal }) {
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (value / 100) * circumference;
  const color = isReal ? '#10B981' : '#EF4444';

  return (
    <div className="relative flex items-center justify-center">
      <svg width="120" height="120" viewBox="0 0 120 120" className="-rotate-90">
        {/* Background Circle */}
        <circle
          cx="60"
          cy="60"
          r={radius}
          fill="none"
          stroke="#334155"
          strokeWidth="8"
        />
        {/* Progress Circle */}
        <circle
          cx="60"
          cy="60"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          className="transition-all duration-1000 ease-out"
          style={{ animation: 'gauge-fill 1s ease-out' }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-bold" style={{ color }}>
          {value.toFixed(1)}%
        </span>
        <span className="text-xs text-ds-text-muted uppercase tracking-wide">
          Confidence
        </span>
      </div>
    </div>
  );
}

// Status Badge Component
function StatusBadge({ isReal, riskLevel }) {
  const config = isReal
    ? {
        icon: ShieldCheck,
        text: 'AUTHENTIC',
        bg: 'bg-ds-success/20',
        border: 'border-ds-success/50',
        textColor: 'text-ds-success',
      }
    : {
        icon: ShieldAlert,
        text: 'LIKELY FAKE',
        bg: 'bg-ds-danger/20',
        border: 'border-ds-danger/50',
        textColor: 'text-ds-danger',
      };

  const Icon = config.icon;

  return (
    <div
      className={`flex items-center gap-2 px-4 py-2 rounded-lg border ${config.bg} ${config.border}`}
    >
      <Icon className={`w-5 h-5 ${config.textColor}`} />
      <span className={`font-semibold text-sm ${config.textColor}`}>
        {config.text}
      </span>
    </div>
  );
}

// Risk Level Indicator
function RiskIndicator({ level }) {
  const colors = {
    low: { bg: 'bg-ds-success', text: 'Low Risk' },
    medium: { bg: 'bg-yellow-500', text: 'Medium Risk' },
    high: { bg: 'bg-orange-500', text: 'High Risk' },
    critical: { bg: 'bg-ds-danger', text: 'Critical' },
  };

  const config = colors[level] || colors.medium;

  return (
    <div className="flex items-center gap-2">
      <div className={`w-3 h-3 rounded-full ${config.bg} animate-pulse-status`} />
      <span className="text-sm text-ds-text-secondary">{config.text}</span>
    </div>
  );
}

export default function Home() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState(null);
  const [fourierUrl, setFourierUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('original');
  const [error, setError] = useState(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState(50);
  const fileInputRef = useRef(null);

  // Dynamically calculate if image is Real based on threshold
  const computedIsReal = result ? result.confidence * 100 > confidenceThreshold : false;

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      // Validate file type
      if (!selectedFile.type.startsWith('image/')) {
        setError('Please select a valid image file.');
        return;
      }
      // Validate file size (max 10MB)
      if (selectedFile.size > 10 * 1024 * 1024) {
        setError('File size must be less than 10MB.');
        return;
      }
      
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      setResult(null);
      setHeatmapUrl(null);
      setFourierUrl(null);
      setError(null);
      setActiveTab('original');
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      // Call the /analyze endpoint using axios
      const response = await axios.post('http://127.0.0.1:8000/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const data = response.data;
      
      setResult({
        isReal: data.is_real,
        confidence: data.confidence_score,
        riskLevel: data.risk_level,
        filename: data.filename,
      });

      // Handle base64 images if present in the response
      if (data.heatmap) {
        setHeatmapUrl(`data:image/png;base64,${data.heatmap}`);
      }
      if (data.fourier) {
        setFourierUrl(`data:image/png;base64,${data.fourier}`);
      }
    } catch (err) {
      if (axios.isAxiosError(err)) {
        const message = err.response?.data?.detail || err.message || 'Analysis failed. Please try again.';
        setError(message);
      } else {
        setError(err.message || 'Analysis failed. Please try again.');
      }
      console.error('Analysis error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setHeatmapUrl(null);
    setFourierUrl(null);
    setError(null);
    setActiveTab('original');
    setConfidenceThreshold(50);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="h-screen w-full overflow-hidden flex bg-ds-primary">
      {/* Left Sidebar */}
      <aside className="w-64 flex-shrink-0 bg-ds-secondary border-r border-ds-border overflow-y-auto">
        <div className="p-4 space-y-6">
          {/* Logo/Brand */}
          <div className="flex items-center gap-3 pb-4 border-b border-ds-border">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-ds-accent to-blue-700 flex items-center justify-center">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-ds-text">DeepScan</h1>
              <p className="text-xs text-ds-text-muted">AI Deepfake Detector</p>
            </div>
          </div>

          {/* Upload Section */}
          <div className="space-y-3">
            <h2 className="text-sm font-semibold text-ds-text-secondary uppercase tracking-wide">
              Upload Image
            </h2>
            
            {/* Hidden File Input */}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
            />

            {/* Upload Button */}
            <button
              onClick={() => fileInputRef.current?.click()}
              className="w-full p-6 border-2 border-dashed border-ds-border rounded-xl hover:border-ds-accent hover:bg-ds-accent/5 transition-all duration-200 group"
            >
              <div className="flex flex-col items-center gap-2">
                <Upload className="w-8 h-8 text-ds-text-muted group-hover:text-ds-accent transition-colors" />
                <span className="text-sm text-ds-text-secondary group-hover:text-ds-text transition-colors">
                  Click to upload
                </span>
                <span className="text-xs text-ds-text-muted">
                  PNG, JPG up to 10MB
                </span>
              </div>
            </button>

            {/* Selected File Info */}
            {file && (
              <div className="p-3 bg-ds-elevated/50 rounded-lg border border-ds-border">
                <div className="flex items-center gap-2">
                  <ImageIcon className="w-4 h-4 text-ds-accent" />
                  <span className="text-sm text-ds-text truncate flex-1">
                    {file.name}
                  </span>
                </div>
                <span className="text-xs text-ds-text-muted">
                  {(file.size / 1024).toFixed(1)} KB
                </span>
              </div>
            )}

            {/* Action Buttons */}
            {file && (
              <div className="space-y-2">
                <button
                  onClick={handleAnalyze}
                  disabled={isLoading}
                  className="w-full py-3 px-4 bg-ds-accent hover:bg-blue-600 disabled:bg-ds-elevated disabled:cursor-not-allowed rounded-lg font-semibold text-white transition-colors flex items-center justify-center gap-2"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Zap className="w-5 h-5" />
                      Analyze Image
                    </>
                  )}
                </button>
                <button
                  onClick={handleClear}
                  disabled={isLoading}
                  className="w-full py-2 px-4 bg-ds-elevated hover:bg-ds-elevated/80 disabled:cursor-not-allowed rounded-lg text-sm text-ds-text-secondary transition-colors"
                >
                  Clear
                </button>
              </div>
            )}
          </div>

          {/* Model Info */}
          <div className="space-y-3 pt-4 border-t border-ds-border">
            <h2 className="text-sm font-semibold text-ds-text-secondary uppercase tracking-wide">
              Model Info
            </h2>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-ds-text-muted">Model</span>
                <span className="text-ds-text">{MODEL_INFO.name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-ds-text-muted">Dataset</span>
                <span className="text-ds-text">{MODEL_INFO.dataset}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-ds-text-muted">Accuracy</span>
                <span className="text-ds-success font-medium">
                  {MODEL_INFO.accuracy}%
                </span>
              </div>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Workspace */}
      <main className="flex-1 overflow-y-auto p-6">
        <div className="h-full flex flex-col">
          {/* Tab Navigation (when image loaded) */}
          {previewUrl && (
            <div className="flex gap-2 mb-4">
              <button
                onClick={() => setActiveTab('original')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
                  activeTab === 'original'
                    ? 'bg-ds-accent text-white'
                    : 'bg-ds-secondary text-ds-text-secondary hover:bg-ds-elevated'
                }`}
              >
                <ImageIcon className="w-4 h-4" />
                Original
              </button>
              {heatmapUrl && (
                <button
                  onClick={() => setActiveTab('heatmap')}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
                    activeTab === 'heatmap'
                      ? 'bg-ds-accent text-white'
                      : 'bg-ds-secondary text-ds-text-secondary hover:bg-ds-elevated'
                  }`}
                >
                  <Eye className="w-4 h-4" />
                  Heatmap
                </button>
              )}
              {fourierUrl && (
                <button
                  onClick={() => setActiveTab('fourier')}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
                    activeTab === 'fourier'
                      ? 'bg-ds-accent text-white'
                      : 'bg-ds-secondary text-ds-text-secondary hover:bg-ds-elevated'
                  }`}
                >
                  <Waves className="w-4 h-4" />
                  Fourier
                </button>
              )}
            </div>
          )}

          {/* Image Display Area */}
          <div className="flex-1 bg-ds-secondary rounded-xl border border-ds-border flex items-center justify-center overflow-hidden">
            {!previewUrl ? (
              // Empty State
              <div className="text-center p-8">
                <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-ds-elevated flex items-center justify-center">
                  <ImageIcon className="w-10 h-10 text-ds-text-muted" />
                </div>
                <h3 className="text-lg font-medium text-ds-text mb-2">
                  No Image Selected
                </h3>
                <p className="text-sm text-ds-text-muted max-w-xs">
                  Upload an image using the sidebar to begin deepfake analysis
                </p>
              </div>
            ) : (
              // Image Display
              <div className="w-full h-full p-4 flex items-center justify-center">
                <img
                  src={
                    activeTab === 'original'
                      ? previewUrl
                      : activeTab === 'heatmap'
                      ? heatmapUrl
                      : fourierUrl
                  }
                  alt={`${activeTab} view`}
                  className="max-w-full max-h-full object-contain rounded-lg shadow-xl"
                />
              </div>
            )}
          </div>

          {/* Error Display */}
          {error && (
            <div className="mt-4 p-4 bg-ds-danger/20 border border-ds-danger/50 rounded-lg flex items-center gap-3">
              <AlertTriangle className="w-5 h-5 text-ds-danger flex-shrink-0" />
              <span className="text-sm text-ds-danger">{error}</span>
            </div>
          )}
        </div>
      </main>

      {/* Right Panel - Analysis Report */}
      <aside className="w-80 flex-shrink-0 bg-ds-secondary border-l border-ds-border overflow-y-auto">
        <div className="p-4 space-y-6">
          {/* Panel Header */}
          <div className="flex items-center gap-2 pb-4 border-b border-ds-border">
            <Activity className="w-5 h-5 text-ds-accent" />
            <h2 className="text-lg font-semibold text-ds-text">
              Analysis Report
            </h2>
          </div>

          {!result ? (
            // Empty State
            <div className="text-center py-12">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-ds-elevated flex items-center justify-center">
                <Shield className="w-8 h-8 text-ds-text-muted" />
              </div>
              <h3 className="text-sm font-medium text-ds-text mb-2">
                No Analysis Yet
              </h3>
              <p className="text-xs text-ds-text-muted">
                Upload and analyze an image to see results
              </p>
            </div>
          ) : (
            // Results Display
            <div className="space-y-6">
              {/* Status Badge - uses dynamic threshold */}
              <div className="flex justify-center">
                <StatusBadge isReal={computedIsReal} riskLevel={result.riskLevel} />
              </div>

              {/* Confidence Threshold Slider */}
              <div className="bg-ds-elevated/30 rounded-xl p-4 border border-ds-border">
                <h3 className="text-sm font-medium text-ds-text-secondary mb-3">
                  Confidence Threshold
                </h3>
                <div className="space-y-2">
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={confidenceThreshold}
                    onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
                    className="w-full h-2 bg-ds-elevated rounded-lg appearance-none cursor-pointer accent-ds-accent"
                  />
                  <div className="flex justify-between text-xs text-ds-text-muted">
                    <span>0%</span>
                    <span className="text-ds-accent font-semibold">{confidenceThreshold}%</span>
                    <span>100%</span>
                  </div>
                  <p className="text-xs text-ds-text-muted mt-2">
                    If confidence &gt; {confidenceThreshold}% → <span className="text-ds-success">Real</span>, otherwise → <span className="text-ds-danger">Fake</span>
                  </p>
                </div>
              </div>

              {/* Confidence Gauge */}
              <div className="bg-ds-elevated/30 rounded-xl p-4 border border-ds-border">
                <h3 className="text-sm font-medium text-ds-text-secondary mb-4 text-center">
                  Detection Confidence
                </h3>
                <div className="flex justify-center">
                  <GaugeChart value={result.confidence * 100} isReal={computedIsReal} />
                </div>
              </div>

              {/* Risk Level */}
              <div className="bg-ds-elevated/30 rounded-xl p-4 border border-ds-border">
                <h3 className="text-sm font-medium text-ds-text-secondary mb-3">
                  Risk Assessment
                </h3>
                <RiskIndicator level={result.riskLevel} />
              </div>

              {/* Forensic Visualizations */}
              {(heatmapUrl || fourierUrl) && (
                <div className="bg-ds-elevated/30 rounded-xl p-4 border border-ds-border space-y-3">
                  <h3 className="text-sm font-medium text-ds-text-secondary">
                    Forensic Analysis
                  </h3>
                  <div className="space-y-3">
                    {heatmapUrl && (
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <Eye className="w-4 h-4 text-ds-accent" />
                          <span className="text-xs text-ds-text-muted">Heatmap</span>
                        </div>
                        <img
                          src={heatmapUrl}
                          alt="Heatmap Analysis"
                          className="w-full rounded-lg border border-ds-border"
                        />
                      </div>
                    )}
                    {fourierUrl && (
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <Waves className="w-4 h-4 text-ds-accent" />
                          <span className="text-xs text-ds-text-muted">Fourier Analysis</span>
                        </div>
                        <img
                          src={fourierUrl}
                          alt="Fourier Analysis"
                          className="w-full rounded-lg border border-ds-border"
                        />
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Analysis Details */}
              <div className="bg-ds-elevated/30 rounded-xl p-4 border border-ds-border space-y-3">
                <h3 className="text-sm font-medium text-ds-text-secondary">
                  Details
                </h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-ds-text-muted">File</span>
                    <span className="text-ds-text truncate max-w-[140px]" title={result.filename}>
                      {result.filename}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-ds-text-muted">Classification</span>
                    <span className={computedIsReal ? 'text-ds-success' : 'text-ds-danger'}>
                      {computedIsReal ? 'Real' : 'Fake'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-ds-text-muted">Score</span>
                    <span className="text-ds-text">
                      {(result.confidence * 100).toFixed(2)}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Interpretation */}
              <div className="bg-ds-elevated/30 rounded-xl p-4 border border-ds-border">
                <div className="flex items-start gap-2">
                  <Info className="w-4 h-4 text-ds-accent mt-0.5 flex-shrink-0" />
                  <p className="text-xs text-ds-text-muted leading-relaxed">
                    {computedIsReal ? (
                      <>
                        This image appears to be <strong className="text-ds-success">authentic</strong>. 
                        The CNN model found patterns consistent with real photographs.
                      </>
                    ) : (
                      <>
                        This image shows signs of <strong className="text-ds-danger">manipulation</strong>. 
                        The analysis detected patterns commonly associated with AI-generated content.
                      </>
                    )}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}
