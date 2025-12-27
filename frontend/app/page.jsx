'use client';

import { useState, useRef, useEffect } from 'react';
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
  Zap,
  ChevronDown,
  Cpu
} from 'lucide-react';

// API Base URL: Uses environment variable in production, localhost in development
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Model Selector Component
function ModelSelector({ models, currentModel, onSelect, isLoading }) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);
  
  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [isOpen]);
  
  const selectedModel = models.find(m => m.id === currentModel) || models[0];
  
  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        disabled={isLoading}
        className="w-full px-3 py-2 bg-ds-elevated border border-ds-border rounded-lg flex items-center justify-between hover:bg-ds-elevated/80 transition-colors disabled:opacity-50"
      >
        <div className="flex items-center gap-2">
          <div 
            className="w-2 h-2 rounded-full" 
            style={{ backgroundColor: selectedModel?.color || '#3B82F6' }}
          />
          <span className="text-sm text-ds-text truncate">
            {selectedModel?.name || 'Select Model'}
          </span>
        </div>
        <ChevronDown className={`w-4 h-4 text-ds-text-muted transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>
      
      {isOpen && (
        <div className="absolute top-full left-0 right-0 mt-1 bg-ds-elevated border border-ds-border rounded-lg shadow-2xl z-[9999] overflow-hidden max-h-64 overflow-y-auto">
          {models.map((model) => (
            <button
              key={model.id}
              onClick={() => {
                onSelect(model.id);
                setIsOpen(false);
              }}
              disabled={!model.is_available}
              className={`w-full px-3 py-2 text-left hover:bg-ds-primary/50 transition-colors flex items-center gap-2 ${
                model.id === currentModel ? 'bg-ds-accent/20' : ''
              } ${!model.is_available ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              <div 
                className="w-2 h-2 rounded-full flex-shrink-0" 
                style={{ backgroundColor: model.color }}
              />
              <div className="flex-1 min-w-0">
                <p className="text-sm text-ds-text truncate">{model.name}</p>
                <p className="text-xs text-ds-text-muted">
                  {model.accuracy}% accuracy • {model.dataset}
                </p>
              </div>
              {model.id === currentModel && (
                <CheckCircle className="w-4 h-4 text-ds-accent flex-shrink-0" />
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

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
  const fileInputRef = useRef(null);
  
  // Model selection state
  const [models, setModels] = useState([]);
  const [currentModel, setCurrentModel] = useState('');
  const [isModelLoading, setIsModelLoading] = useState(false);

  // Fetch available models on mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        console.log('🔍 Fetching models from:', `${API_BASE}/models`);
        const response = await axios.get(`${API_BASE}/models`);
        console.log('✅ Models fetched:', response.data);
        setModels(response.data.models);
        setCurrentModel(response.data.current_model);
      } catch (err) {
        console.error('❌ Failed to fetch models:', err);
        console.error('API Base:', API_BASE);
        // Set default fallback
        setModels([{
          id: 'cnn_cifake',
          name: 'CNN Custom (CIFAKE)',
          accuracy: 92.0,
          dataset: 'CIFAKE',
          color: '#10B981',
          is_available: true,
          is_current: true
        }]);
        setCurrentModel('cnn_cifake');
      }
    };
    fetchModels();
  }, []);

  // Handle model selection change
  const handleModelSelect = async (modelId) => {
    if (modelId === currentModel) return;
    
    setIsModelLoading(true);
    try {
      const response = await axios.post(`${API_BASE}/models/${modelId}/select`);
      if (response.data.success) {
        setCurrentModel(modelId);
        // Clear previous results when model changes
        setResult(null);
        setHeatmapUrl(null);
        setFourierUrl(null);
      }
    } catch (err) {
      console.error('Failed to select model:', err);
      setError('Failed to switch model. Please try again.');
    } finally {
      setIsModelLoading(false);
    }
  };

  // Get current model info
  const currentModelInfo = models.find(m => m.id === currentModel) || {};

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
      const response = await axios.post(`${API_BASE}/analyze`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 180000, // 180 second timeout for cold starts on Render
      });

      const data = response.data;
      
      setResult({
        isReal: data.is_real,
        confidence: data.confidence_score,
        riskLevel: data.risk_level,
        filename: data.filename,
        // Hybrid analysis fields
        finalScore: data.final_score,
        heatmapScore: data.heatmap_score,
        fourierScore: data.fourier_score,
        analysisReason: data.analysis_reason,
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
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="h-screen w-full overflow-hidden flex bg-ds-primary relative">
      {/* Student Info Badge */}
      <div className="absolute top-2 left-1/2 -translate-x-1/2 z-50 px-4 py-2 rounded-lg bg-ds-secondary/80 backdrop-blur-md border border-ds-border shadow-lg">
        <p className="text-sm font-medium text-ds-text">Nguyễn Đức Thịnh - 23110156</p>
      </div>

      {/* Left Sidebar */}
      <aside className="w-64 flex-shrink-0 bg-ds-secondary border-r border-ds-border overflow-y-auto relative">
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
          <div className="space-y-3 pt-4 border-t border-ds-border relative z-50">
            <h2 className="text-sm font-semibold text-ds-text-secondary uppercase tracking-wide flex items-center gap-2">
              <Cpu className="w-4 h-4" />
              Select Model
            </h2>
            
            {/* Model Selector Dropdown - with explicit positioning */}
            <div className="relative overflow-visible">
              <ModelSelector
                models={models}
                currentModel={currentModel}
                onSelect={handleModelSelect}
                isLoading={isModelLoading || isLoading}
              />
            </div>
            
            {/* Current Model Info */}
            {currentModelInfo.name && (
              <div className="space-y-2 text-sm p-3 bg-ds-elevated/30 rounded-lg">
                <div className="flex justify-between">
                  <span className="text-ds-text-muted">Architecture</span>
                  <span className="text-ds-text">{currentModelInfo.architecture || 'CNN'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-ds-text-muted">Dataset</span>
                  <span className="text-ds-text">{currentModelInfo.dataset || 'Unknown'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-ds-text-muted">Accuracy</span>
                  <span className="text-ds-success font-medium">
                    {currentModelInfo.accuracy || 0}%
                  </span>
                </div>
              </div>
            )}
            
            {isModelLoading && (
              <div className="flex items-center gap-2 text-xs text-ds-text-muted">
                <Loader2 className="w-3 h-3 animate-spin" />
                Loading model...
              </div>
            )}
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
              {/* Status Badge - uses hybrid analysis result from backend */}
              <div className="flex justify-center">
                <StatusBadge isReal={result.isReal} riskLevel={result.riskLevel} />
              </div>

              {/* Hybrid Score Explanation */}
              <div className="bg-ds-elevated/30 rounded-xl p-4 border border-ds-border">
                <h3 className="text-sm font-medium text-ds-text-secondary mb-3">
                  Hybrid Analysis Mode
                </h3>
                <p className="text-xs text-ds-text-muted">
                  Final score = 40% CNN + 30% Heatmap + 30% Fourier. 
                  Forensic analysis can override CNN if anomalies detected.
                </p>
              </div>

              {/* Confidence Gauge - shows Final Hybrid Score */}
              <div className="bg-ds-elevated/30 rounded-xl p-4 border border-ds-border">
                <h3 className="text-sm font-medium text-ds-text-secondary mb-4 text-center">
                  Final Hybrid Score
                </h3>
                <div className="flex justify-center">
                  <GaugeChart value={result.confidence * 100} isReal={result.isReal} />
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
                          {result.heatmapScore !== undefined && (
                            <span className={`text-xs font-medium ${result.heatmapScore > 0.5 ? 'text-ds-danger' : 'text-ds-success'}`}>
                              ({(result.heatmapScore * 100).toFixed(1)}%)
                            </span>
                          )}
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
                          {result.fourierScore !== undefined && (
                            <span className={`text-xs font-medium ${result.fourierScore > 0.5 ? 'text-ds-danger' : 'text-ds-success'}`}>
                              ({(result.fourierScore * 100).toFixed(1)}%)
                            </span>
                          )}
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
                    <span className={result.isReal ? 'text-ds-success' : 'text-ds-danger'}>
                      {result.isReal ? 'Real' : 'Fake'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-ds-text-muted">Final Score</span>
                    <span className="text-ds-text">
                      {(result.confidence * 100).toFixed(2)}%
                    </span>
                  </div>
                  {result.heatmapScore !== undefined && (
                    <div className="flex justify-between">
                      <span className="text-ds-text-muted">Heatmap Score</span>
                      <span className={result.heatmapScore > 0.5 ? 'text-ds-danger' : 'text-ds-success'}>
                        {(result.heatmapScore * 100).toFixed(1)}%
                      </span>
                    </div>
                  )}
                  {result.fourierScore !== undefined && (
                    <div className="flex justify-between">
                      <span className="text-ds-text-muted">Fourier Score</span>
                      <span className={result.fourierScore > 0.5 ? 'text-ds-danger' : 'text-ds-success'}>
                        {(result.fourierScore * 100).toFixed(1)}%
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Analysis Interpretation - Dynamic from Backend */}
              <div className="bg-ds-elevated/30 rounded-xl p-4 border border-ds-border">
                <div className="flex items-start gap-2">
                  <Info className="w-4 h-4 text-ds-accent mt-0.5 flex-shrink-0" />
                  <div className="text-xs text-ds-text-muted leading-relaxed space-y-2">
                    {result.analysisReason ? (
                      // Parse and display each reason on a new line
                      result.analysisReason.split(' | ').map((reason, index) => (
                        <p key={index} className={
                          reason.includes('✓') ? 'text-ds-success' :
                          reason.includes('❌') || reason.includes('⚠️') ? 'text-ds-danger' :
                          reason.startsWith('[') ? 'text-ds-text-muted font-mono text-[10px]' :
                          'text-ds-text-secondary'
                        }>
                          {reason}
                        </p>
                      ))
                    ) : (
                      // Fallback static message
                      <p>
                        {result.isReal ? (
                          <>
                            This image appears to be <strong className="text-ds-success">authentic</strong>. 
                            The analysis found patterns consistent with real photographs.
                          </>
                        ) : (
                          <>
                            This image shows signs of <strong className="text-ds-danger">manipulation</strong>. 
                            The analysis detected patterns commonly associated with AI-generated content.
                          </>
                        )}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}
