import { useState } from 'react';
import { Send, AlertCircle, CheckCircle, Loader, Activity, LogOut, User, Moon, Sun, Dna, FlaskConical, Download, Sparkles } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface User {
  id: string;
  username: string;
}

interface PredictionResult {
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
}

interface GeneratedSequence {
  sequence: string;
  average_probability: number;
  levenshtein: number;
  hamming: number;
}

interface ApiResponse {
  success: boolean;
  result: PredictionResult;
  processed_sequence: string;
  error?: string;
}

interface SequenceApiResponse {
  success: boolean;
  input_sequence: string;
  total_generated?: number;
  top_sequences: GeneratedSequence[];
  metrics_used?: string[];
  ranking?: string;
  error?: string;
}

interface SmilesApiResponse {
  success: boolean;
  input_sequence: string;
  smiles: string;
  error?: string;
}

export default function DashboardPage({ user, onLogout, isDarkMode, toggleTheme }: {
  user: User | null;
  onLogout: () => void;
  isDarkMode: boolean;
  toggleTheme: () => void;
}) {
  const navigate = useNavigate();

  if (!user) {
    navigate('/');
    return null;
  }
  const [sequence, setSequence] = useState('');
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [generatorSequence, setGeneratorSequence] = useState('');
  const [sequenceGenerationLoading, setSequenceGenerationLoading] = useState(false);
  const [generatedSequences, setGeneratedSequences] = useState<GeneratedSequence[]>([]);
  const [sequenceGenerationError, setSequenceGenerationError] = useState<string | null>(null);
  const [smilesSequence, setSmilesSequence] = useState('');
  const [smilesLoading, setSmilesLoading] = useState(false);
  const [generatedSmiles, setGeneratedSmiles] = useState<string>('');
  const [smilesError, setSmilesError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'classification' | 'sequence-generation' | 'smiles-generation' | 'alphafold2' | 'docking'>('classification');
  
  // AlphaFold2 state
  const [alphafoldSequence, setAlphafoldSequence] = useState('');
  const [alphafoldLoading, setAlphafoldLoading] = useState(false);
  const [alphafoldResult, setAlphafoldResult] = useState<any>(null);
  const [alphafoldError, setAlphafoldError] = useState<string | null>(null);
  
  // Docking state
  const [dockingSmiles, setDockingSmiles] = useState('');
  const [dockingProteinPDB, setDockingProteinPDB] = useState('');
  const [dockingLoading, setDockingLoading] = useState(false);
  const [dockingResult, setDockingResult] = useState<any>(null);
  const [dockingError, setDockingError] = useState<string | null>(null);
  const [useProteinSequence, setUseProteinSequence] = useState(false);
  const [dockingProteinSequence, setDockingProteinSequence] = useState('');

  const validateSequence = (seq: string): boolean => {
    const validAminoAcids = /^[ACDEFGHIKLMNPQRSTVWY]+$/i;
    return validAminoAcids.test(seq.trim());
  };

  const handlePredict = async () => {
    const trimmedSequence = sequence.trim().toUpperCase();
    
    if (!trimmedSequence) {
      setError('Please enter a protein sequence');
      return;
    }
    
    if (trimmedSequence.length < 10) {
      setError('Sequence too short. Please provide at least 10 amino acids.');
      return;
    }
    
    if (!validateSequence(trimmedSequence)) {
      setError('Invalid amino acid sequence. Please use single-letter amino acid codes (A-Z, excluding B, J, O, U, X, Z).');
      return;
    }

    setPredictionLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:5001/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ sequence: trimmedSequence }),
      });

      if (response.status === 401) {
        setError('Session expired. Please log in again.');
        return;
      }

      const data: ApiResponse = await response.json();

      if (data.success && data.result) {
        setResult(data.result);
      } else {
        setError(data.error || 'Prediction failed');
      }
    } catch (err) {
      console.error('Prediction error:', err);
      setError('Unable to connect to the server. Please ensure the Flask backend is running.');
    } finally {
      setPredictionLoading(false);
    }
  };

  const handleGenerateSequences = async () => {
    const trimmedSequence = generatorSequence.trim().toUpperCase();
    
    if (!trimmedSequence) {
      setSequenceGenerationError('Please enter a protein sequence');
      return;
    }
    
    if (trimmedSequence.length < 10) {
      setSequenceGenerationError('Sequence too short. Please provide at least 10 amino acids.');
      return;
    }
    
    if (!validateSequence(trimmedSequence)) {
      setSequenceGenerationError('Invalid amino acid sequence. Please use single-letter amino acid codes (A-Z, excluding B, J, O, U, X, Z).');
      return;
    }

    setSequenceGenerationLoading(true);
    setSequenceGenerationError(null);
    setGeneratedSequences([]);

    try {
      const response = await fetch('http://localhost:5001/generate-sequences', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ sequence: trimmedSequence }),
      });

      if (response.status === 401) {
        setSequenceGenerationError('Session expired. Please log in again.');
        return;
      }

      const data: SequenceApiResponse = await response.json();

      if (data.success && data.top_sequences) {
        setGeneratedSequences(data.top_sequences);
      } else {
        setSequenceGenerationError(data.error || 'Sequence generation failed');
      }
    } catch (err) {
      console.error('Sequence generation error:', err);
      setSequenceGenerationError('Unable to connect to the server. Please ensure the Flask backend is running.');
    } finally {
      setSequenceGenerationLoading(false);
    }
  };

  const handleGenerateSmiles = async () => {
    const trimmedSequence = smilesSequence.trim().toUpperCase();
    
    if (!trimmedSequence) {
      setSmilesError('Please enter a protein sequence');
      return;
    }
    
    if (trimmedSequence.length < 10) {
      setSmilesError('Sequence too short. Please provide at least 10 amino acids.');
      return;
    }
    
    if (!validateSequence(trimmedSequence)) {
      setSmilesError('Invalid amino acid sequence. Please use single-letter amino acid codes (A-Z, excluding B, J, O, U, X, Z).');
      return;
    }

    setSmilesLoading(true);
    setSmilesError(null);
    setGeneratedSmiles('');

    try {
      const response = await fetch('http://localhost:5001/generate-smiles', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ sequence: trimmedSequence }),
      });

      if (response.status === 401) {
        setSmilesError('Session expired. Please log in again.');
        return;
      }

      const data: SmilesApiResponse = await response.json();

      if (data.success && data.smiles) {
        setGeneratedSmiles(data.smiles);
      } else {
        setSmilesError(data.error || 'SMILES generation failed');
      }
    } catch (err) {
      console.error('SMILES generation error:', err);
      setSmilesError('Unable to connect to the server. Please ensure the Flask backend is running.');
    } finally {
      setSmilesLoading(false);
    }
  };

  const getSeverityColor = (prediction: string): string => {
    const severityColors: Record<string, string> = {
      'Pathogenic': 'text-red-500',
      'Likely_pathogenic': 'text-orange-500',
      'Uncertain_significance': 'text-yellow-500',
      'Likely_benign': 'text-blue-500',
      'Benign': 'text-green-500',
    };
    return severityColors[prediction] || 'text-gray-500';
  };

  const formatPrediction = (prediction: string): string => {
    return prediction.replace(/_/g, ' ');
  };

  // AlphaFold2 handler
  const handleAlphaFold2Predict = async () => {
    const trimmedSequence = alphafoldSequence.trim().toUpperCase();
    
    if (!trimmedSequence) {
      setAlphafoldError('Please enter a protein sequence');
      return;
    }
    
    if (trimmedSequence.length < 10) {
      setAlphafoldError('Sequence too short. Please provide at least 10 amino acids.');
      return;
    }
    
    if (!validateSequence(trimmedSequence)) {
      setAlphafoldError('Invalid amino acid sequence. Please use single-letter amino acid codes.');
      return;
    }

    setAlphafoldLoading(true);
    setAlphafoldError(null);
    setAlphafoldResult(null);

    try {
      // Create AbortController for timeout handling
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 900000); // 15 minutes timeout
      
      const response = await fetch('http://localhost:5001/alphafold2/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ sequence: trimmedSequence }),
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);

      if (response.status === 401) {
        setAlphafoldError('Session expired. Please log in again.');
        return;
      }

      const data = await response.json();

      if (data.success) {
        setAlphafoldResult(data);
      } else {
        setAlphafoldError(data.error || 'AlphaFold2 prediction failed');
      }
    } catch (err: any) {
      console.error('AlphaFold2 error:', err);
      if (err.name === 'AbortError' || err.message?.includes('timeout')) {
        setAlphafoldError('Request timeout. AlphaFold2 predictions can take 5-15 minutes. Please try again with a shorter sequence for testing.');
      } else {
        setAlphafoldError('Unable to connect to the server. Please ensure the Flask backend and Colab notebook (via ngrok) are running.');
      }
    } finally {
      setAlphafoldLoading(false);
    }
  };

  const downloadAlphaFoldPDB = () => {
    if (alphafoldResult?.result?.pdb_content) {
      const blob = new Blob([alphafoldResult.result.pdb_content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `alphafold2_prediction_${alphafoldResult.sequence.substring(0, 10)}.pdb`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  // Docking handler
  const handleDocking = async () => {
    if (!dockingSmiles.trim()) {
      setDockingError('Please enter a SMILES string');
      return;
    }
    
    if (!useProteinSequence && !dockingProteinPDB.trim()) {
      setDockingError('Please provide protein PDB content (from AlphaFold2) or sequence');
      return;
    }

    if (useProteinSequence && !dockingProteinSequence.trim()) {
      setDockingError('Please enter a protein sequence');
      return;
    }

    setDockingLoading(true);
    setDockingError(null);
    setDockingResult(null);

    try {
      const payload: any = {
        smiles: dockingSmiles.trim(),
      };

      if (useProteinSequence) {
        payload.protein_sequence = dockingProteinSequence.trim().toUpperCase();
      } else {
        payload.protein_pdb = dockingProteinPDB.trim();
      }

      const response = await fetch('http://localhost:5001/docking/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify(payload),
      });

      if (response.status === 401) {
        setDockingError('Session expired. Please log in again.');
        return;
      }

      const data = await response.json();

      if (data.success) {
        setDockingResult(data);
      } else {
        setDockingError(data.error || 'Docking failed');
      }
    } catch (err) {
      console.error('Docking error:', err);
      setDockingError('Unable to connect to the server. Please ensure the Flask backend and Colab notebook (via ngrok) are running.');
    } finally {
      setDockingLoading(false);
    }
  };

  const downloadDockingPDBQT = () => {
    if (dockingResult?.result?.pdbqt_content) {
      const blob = new Blob([dockingResult.result.pdbqt_content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `docked_poses_${dockingResult.smiles.substring(0, 10).replace(/[^a-zA-Z0-9]/g, '_')}.pdbqt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  return (
    <div className={`min-h-screen w-full max-w-full overflow-x-hidden relative flex flex-col transition-all duration-500 ease-in-out flex-1 ${isDarkMode ? 'bg-gradient-to-br from-slate-900 via-blue-950 to-indigo-950' : 'bg-gradient-to-br from-blue-900 via-indigo-900 to-purple-900'}`}>
      {/* Background particles effect */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className={`absolute top-20 left-20 w-2 h-2 rounded-full opacity-20 animate-pulse transition-colors duration-500 ${isDarkMode ? 'bg-blue-400' : 'bg-white'}`}></div>
        <div className={`absolute top-40 right-32 w-1 h-1 rounded-full opacity-30 animate-pulse transition-colors duration-500 ${isDarkMode ? 'bg-purple-400' : 'bg-blue-300'}`}></div>
        <div className={`absolute bottom-32 left-16 w-2 h-2 rounded-full opacity-25 animate-pulse transition-colors duration-500 ${isDarkMode ? 'bg-teal-400' : 'bg-purple-300'}`}></div>
      </div>

      <div className="relative z-10 w-full max-w-7xl mx-auto px-4 py-8 flex flex-col items-center flex-1">
        {/* Header with user info, theme toggle, and logout */}
        <div className="absolute top-4 right-4 flex items-center gap-4">
          <div className={`flex items-center gap-2 backdrop-blur-sm rounded-full px-4 py-2 border transition-all duration-300 ${isDarkMode ? 'bg-slate-800/60 border-blue-600 shadow-lg' : 'bg-white/10 border-white/20'}`}>
            <User className={`w-4 h-4 transition-colors duration-300 ${isDarkMode ? 'text-blue-300' : 'text-blue-300'}`} />
            <span className={`text-sm font-medium transition-colors duration-300 ${isDarkMode ? 'text-blue-100' : 'text-white'}`}>{user.username}</span>
          </div>
          <button
            onClick={toggleTheme}
            className={`rounded-full p-2 transition-all duration-300 hover:scale-110 ${isDarkMode ? 'bg-indigo-700 hover:bg-indigo-600 text-indigo-200 shadow-lg' : 'bg-white/10 hover:bg-white/20 text-blue-300'}`}
            title={isDarkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          >
            {isDarkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
          <button
            onClick={onLogout}
            className="bg-red-500 hover:bg-red-600 text-white rounded-full p-2 transition-all duration-300 hover:scale-110 shadow-lg"
            title="Logout"
          >
            <LogOut className="w-4 h-4" />
          </button>
        </div>

        {/* Header */}
        <div className="text-center mb-10 space-y-3">
          <div className="flex items-center justify-center mb-4">
            <div className="p-3.5 rounded-2xl bg-cyan-500/20 border border-cyan-500/40 text-cyan-400 mr-4 shadow-[0_0_20px_rgba(6,182,212,0.3)]">
              <Activity className="w-9 h-9" />
            </div>
            <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight text-white">
              Insulin T2D <span className="gradient-text-cyan">Drug Analysis</span>
            </h1>
          </div>
          <p className="text-slate-300 text-lg font-light max-w-3xl mx-auto">
            Advanced AI-powered protein sequence pathogenicity classifier & 3D folding suite
          </p>
        </div>

        {/* Navigation Bar */}
        <div className="w-full max-w-7xl mb-8">
          <div className="flex flex-wrap justify-center gap-3 bg-slate-900/80 p-2 rounded-2xl border border-slate-700/60 backdrop-blur-md">
            {[
              { id: 'classification', label: 'Classification', color: 'cyan' },
              { id: 'sequence-generation', label: 'Sequence Generation', color: 'purple' },
              { id: 'smiles-generation', label: 'SMILES Generation', color: 'emerald' },
              { id: 'alphafold2', label: 'AlphaFold2', color: 'indigo' },
              { id: 'docking', label: 'Docking', color: 'teal' },
            ].map((tab) => {
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`px-6 py-3 rounded-xl font-bold text-sm transition-all duration-300 ${
                    isActive
                      ? 'bg-gradient-to-r from-cyan-500 to-indigo-600 text-white shadow-[0_0_20px_rgba(6,182,212,0.4)] scale-105'
                      : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
                  }`}
                >
                  {tab.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Main Card - Content */}
        <div className="w-full max-w-7xl">
          {/* Classification Tab Content */}
          {activeTab === 'classification' && (
            <div className="glass-panel rounded-3xl p-8 mb-12 border border-cyan-500/20 shadow-2xl space-y-6">
              <label htmlFor="sequence" className="block text-xl font-bold text-white">Protein Sequence</label>
              <textarea
                id="sequence"
                value={sequence}
                onChange={(e) => setSequence(e.target.value)}
                placeholder="Enter amino acid sequence (e.g., GIVEQCCTSICSLYQLENYCN)..."
                className="w-full h-36 px-6 py-4 rounded-2xl bg-slate-950/80 border border-slate-700 focus:border-cyan-400 text-cyan-300 placeholder-slate-500 font-mono text-sm leading-relaxed outline-none transition-all duration-300 focus:ring-2 focus:ring-cyan-500/20"
                disabled={predictionLoading}
              />
              <div className="mt-4 flex flex-col md:flex-row md:items-center md:gap-6 gap-3">
                <button
                  onClick={handlePredict}
                  disabled={predictionLoading || !sequence.trim()}
                  className="btn-neon-cyan px-8 py-4 rounded-xl font-bold text-white flex items-center justify-center gap-3 transition-all duration-300 text-base"
                >
                  {predictionLoading ? (<><Loader className="w-5 h-5 animate-spin" />Analyzing...</>) : (<><Send className="w-5 h-5" />Predict Pathogenicity</>)}
                </button>
                {error && <div className="flex items-center gap-3 p-4 bg-red-950/60 border border-red-500/40 rounded-xl text-red-300 text-xs animate-fadeIn"><AlertCircle className="w-5 h-5 text-red-400 shrink-0" /><p>{error}</p></div>}
              </div>
              {result && (
                <div className="space-y-6 animate-fadeIn mt-6">
                  <div className="flex items-center gap-3 p-5 bg-green-50 border border-green-200 rounded-xl">
                    <CheckCircle className="w-6 h-6 text-green-500 flex-shrink-0" />
                    <div>
                      <p className="text-green-800 font-semibold text-lg">Prediction Complete</p>
                      <p className="text-green-600 text-sm">Analysis finished successfully</p>
                    </div>
                  </div>

                  <div className={`rounded-xl p-8 space-y-6 transition-all duration-300 ${isDarkMode ? 'bg-slate-700/50 border border-indigo-600' : 'bg-gray-50'}`}>
                    <div className="grid md:grid-cols-2 gap-8">
                      <div className="space-y-3">
                        <h3 className={`text-xl font-bold transition-colors duration-300 ${isDarkMode ? 'text-blue-100' : 'text-gray-800'}`}>Prediction</h3>
                        <p className={`text-3xl font-bold ${getSeverityColor(result.prediction)}`}>
                          {formatPrediction(result.prediction)}
                        </p>
                        <p className={`text-lg transition-colors duration-300 ${isDarkMode ? 'text-blue-200' : 'text-gray-600'}`}>
                          Confidence: <span className="font-semibold">{(result.confidence * 100).toFixed(1)}%</span>
                        </p>
                      </div>

                      <div className="space-y-3">
                        <h3 className={`text-xl font-bold transition-colors duration-300 ${isDarkMode ? 'text-blue-100' : 'text-gray-800'}`}>All Probabilities</h3>
                        <div className="space-y-3">
                          {Object.entries(result.probabilities)
                            .sort(([,a], [,b]) => b - a)
                            .map(([className, probability]) => (
                              <div key={className} className="flex justify-between items-center py-1">
                                <span className={`font-medium transition-colors duration-300 ${isDarkMode ? 'text-blue-200' : 'text-gray-700'}`}>
                                  {formatPrediction(className)}
                                </span>
                                <span className={`font-semibold transition-colors duration-300 ${isDarkMode ? 'text-blue-100' : 'text-gray-800'}`}>
                                  {(probability * 100).toFixed(1)}%
                                </span>
                              </div>
                            ))}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Sequence Generation Tab Content */}
          {activeTab === 'sequence-generation' && (
          <div className="glass-panel rounded-3xl p-8 mb-12 border border-purple-500/20 shadow-2xl space-y-6">
            <label htmlFor="generatorSequence" className="block text-xl font-bold text-white">Input Protein Sequence for Generation</label>
            <textarea
              id="generatorSequence"
              value={generatorSequence}
              onChange={(e) => setGeneratorSequence(e.target.value)}
              placeholder="Enter protein sequence to generate new variants..."
              className="w-full h-32 px-6 py-4 rounded-2xl bg-slate-950/80 border border-slate-700 focus:border-purple-400 text-purple-300 placeholder-slate-500 font-mono text-sm leading-relaxed outline-none transition-all duration-300 focus:ring-2 focus:ring-purple-500/20"
              disabled={sequenceGenerationLoading}
            />
            <div className="mt-4 flex flex-col md:flex-row md:items-center md:gap-6 gap-3">
              <button
                onClick={handleGenerateSequences}
                disabled={sequenceGenerationLoading || !generatorSequence.trim()}
                className="px-8 py-4 rounded-xl font-bold text-white bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-400 hover:to-pink-500 shadow-[0_0_20px_rgba(168,85,247,0.4)] transition-all duration-300 flex items-center justify-center gap-3 text-base disabled:opacity-50"
              >
                {sequenceGenerationLoading ? (<><Loader className="w-5 h-5 animate-spin" />Generating...</>) : (<><Activity className="w-5 h-5" />Generate Sequences</>)}
              </button>
              {sequenceGenerationError && <div className="flex items-center gap-3 p-4 bg-red-950/60 border border-red-500/40 rounded-xl text-red-300 text-xs animate-fadeIn"><AlertCircle className="w-5 h-5 text-red-400 shrink-0" /><p>{sequenceGenerationError}</p></div>}
            </div>
            {generatedSequences.length > 0 && (
                <div className="space-y-6 animate-fadeIn mt-6">
                  <div className="flex items-center gap-3 p-5 bg-purple-950/50 border border-purple-500/30 rounded-2xl">
                    <CheckCircle className="w-6 h-6 text-purple-400 shrink-0" />
                    <div>
                      <p className="text-purple-200 font-bold text-base">Sequence Generation Complete</p>
                      <p className="text-purple-400 text-xs">Generated {generatedSequences.length} sequences with similarity analysis</p>
                    </div>
                  </div>

                  <div className="glass-card-purple rounded-3xl p-8 space-y-6">
                    <div className="text-center">
                      <h3 className="text-2xl font-bold text-white mb-1">
                        Generated Protein Sequences
                      </h3>
                      <p className="text-slate-300 text-sm font-light">
                        Top 10 sequences ranked by average of Levenshtein and Hamming similarities
                      </p>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                      {generatedSequences.map((seq, index) => (
                        <div 
                          key={`seq-${index}`}
                          className="glass-panel p-5 rounded-2xl border border-purple-500/30 hover:border-purple-400 transition-all duration-300 flex flex-col justify-between"
                        >
                          <div className="space-y-3 flex-1 flex flex-col">
                            <div className="flex items-center gap-2 mb-1">
                              <div className={`w-7 h-7 rounded-full flex items-center justify-center text-slate-950 font-black text-xs shrink-0 ${
                                index === 0 ? 'bg-amber-400' : 
                                index === 1 ? 'bg-slate-300' : 
                                index === 2 ? 'bg-orange-400' : 'bg-cyan-400'
                              }`}>
                                {index + 1}
                              </div>
                              <h4 className="text-sm font-bold text-white">
                                Sequence #{index + 1}
                              </h4>
                            </div>
                            
                            <div className="font-mono text-xs p-3 rounded-xl bg-slate-950 text-emerald-400 border border-slate-800 overflow-auto max-h-24 break-all leading-relaxed">
                              {seq.sequence}
                            </div>
                            
                            <div className="text-center p-3 rounded-xl bg-purple-950/60 border border-purple-500/30 mt-auto">
                              <div className="text-xl font-extrabold text-purple-300">
                                {(seq.average_probability * 100).toFixed(1)}%
                              </div>
                              <div className="text-[10px] text-purple-200">
                                Avg Probability
                              </div>
                            </div>
                            
                            <div className="grid grid-cols-2 gap-2 mt-2 text-xs">
                              <div className="p-2 rounded-xl bg-slate-900 text-center border border-slate-800">
                                <div className="font-bold text-cyan-300">
                                  {(seq.levenshtein * 100).toFixed(1)}%
                                </div>
                                <div className="text-[10px] text-slate-400">
                                  Levenshtein
                                </div>
                              </div>
                              <div className="p-2 rounded-xl bg-slate-900 text-center border border-slate-800">
                                <div className="font-bold text-emerald-300">
                                  {(seq.hamming * 100).toFixed(1)}%
                                </div>
                                <div className="text-[10px] text-slate-400">
                                  Hamming
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
          </div>
          )}

          {/* SMILES Generation Tab Content */}
          {activeTab === 'smiles-generation' && (
          <div className="glass-panel rounded-3xl p-8 mb-12 border border-emerald-500/20 shadow-2xl space-y-6">
            <label htmlFor="smilesSequence" className="block text-xl font-bold text-white">Protein Sequence (for SMILES)</label>
            <textarea
              id="smilesSequence"
              value={smilesSequence}
              onChange={(e) => setSmilesSequence(e.target.value)}
              placeholder="Enter protein sequence to generate SMILES..."
              className="w-full h-32 px-6 py-4 rounded-2xl bg-slate-950/80 border border-slate-700 focus:border-emerald-400 text-emerald-300 placeholder-slate-500 font-mono text-sm leading-relaxed outline-none transition-all duration-300 focus:ring-2 focus:ring-emerald-500/20"
              disabled={smilesLoading}
            />
            <div className="mt-4 flex flex-col md:flex-row md:items-center md:gap-6 gap-3">
              <button
                onClick={handleGenerateSmiles}
                disabled={smilesLoading || !smilesSequence.trim()}
                className="px-8 py-4 rounded-xl font-bold text-white bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 shadow-[0_0_20px_rgba(52,211,153,0.4)] transition-all duration-300 flex items-center justify-center gap-3 text-base disabled:opacity-50"
              >
                {smilesLoading ? (<><Loader className="w-5 h-5 animate-spin" />Generating...</>) : (<><Activity className="w-5 h-5" />Generate SMILES</>)}
              </button>
              {smilesError && <div className="flex items-center gap-3 p-4 bg-red-950/60 border border-red-500/40 rounded-xl text-red-300 text-xs animate-fadeIn"><AlertCircle className="w-5 h-5 text-red-400 shrink-0" /><p>{smilesError}</p></div>}
            </div>
            {generatedSmiles && (
                  <div className="space-y-4 animate-fadeIn mt-6">
                    <div className="flex items-center gap-3 p-5 bg-emerald-950/50 border border-emerald-500/30 rounded-2xl">
                      <CheckCircle className="w-6 h-6 text-emerald-400 shrink-0" />
                      <div>
                        <p className="text-emerald-200 font-bold text-base">SMILES Generation Complete</p>
                        <p className="text-emerald-400 text-xs">Generated SMILES chemical structure from input sequence</p>
                      </div>
                    </div>

                    <div className="glass-card-emerald p-6 rounded-3xl border border-emerald-500/30 space-y-3">
                      <h4 className="text-base font-bold text-white">
                        Generated SMILES Structure
                      </h4>
                      <div className="font-mono text-sm p-4 rounded-2xl bg-slate-950 text-emerald-300 border border-slate-800 break-all leading-relaxed">
                        {generatedSmiles}
                      </div>
                    </div>
                  </div>
                )}
          </div>
          )}

          {/* AlphaFold2 Tab Content */}
          {activeTab === 'alphafold2' && (
            <div className="glass-panel rounded-3xl p-8 mb-12 border border-indigo-500/20 shadow-2xl space-y-6">
              <label htmlFor="alphafold-sequence" className="block text-xl font-bold text-white">
                Protein Sequence
              </label>
              <textarea
                id="alphafold-sequence"
                value={alphafoldSequence}
                onChange={(e) => setAlphafoldSequence(e.target.value)}
                placeholder="Enter protein sequence to predict 3D structure..."
                className="w-full h-40 px-6 py-4 rounded-2xl bg-slate-950/80 border border-slate-700 focus:border-indigo-400 text-indigo-300 placeholder-slate-500 font-mono text-sm leading-relaxed outline-none transition-all duration-300 focus:ring-2 focus:ring-indigo-500/20"
                disabled={alphafoldLoading}
              />
              <div className="mt-4 flex flex-col md:flex-row md:items-center md:gap-6 gap-3">
                <button
                  onClick={handleAlphaFold2Predict}
                  disabled={alphafoldLoading || !alphafoldSequence.trim()}
                  className="px-8 py-4 rounded-xl font-bold text-white bg-gradient-to-r from-indigo-500 to-cyan-600 hover:from-indigo-400 hover:to-cyan-500 shadow-[0_0_20px_rgba(99,102,241,0.4)] transition-all duration-300 flex items-center justify-center gap-3 text-base disabled:opacity-50"
                >
                  {alphafoldLoading ? (
                    <>
                      <Loader className="w-5 h-5 animate-spin" />
                      <span>Predicting Structure...</span>
                    </>
                  ) : (
                    <>
                      <Dna className="w-5 h-5" />
                      <span>Predict Structure</span>
                    </>
                  )}
                </button>
                {alphafoldError && (
                  <div className="flex items-center gap-3 p-4 bg-red-950/60 border border-red-500/40 rounded-xl text-red-300 text-xs animate-fadeIn">
                    <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
                    <p className="text-red-700">{alphafoldError}</p>
                  </div>
                )}
              </div>
              {alphafoldResult && alphafoldResult.success && (
                <div className="space-y-6 animate-fadeIn mt-6">
                  <div className="flex items-center gap-3 p-5 bg-green-50 border border-green-200 rounded-xl">
                    <CheckCircle className="w-6 h-6 text-green-500 flex-shrink-0" />
                    <div>
                      <p className="text-green-800 font-semibold text-lg">Structure Prediction Complete</p>
                      <p className="text-green-600 text-sm">AlphaFold2 has generated the 3D structure</p>
                    </div>
                  </div>
                  <div className={`rounded-xl p-8 space-y-6 transition-all duration-300 ${isDarkMode ? 'bg-slate-700/50 border border-indigo-600' : 'bg-indigo-50'}`}>
                    <div className="flex items-center justify-between">
                      <h3 className={`text-2xl font-bold transition-colors duration-300 ${isDarkMode ? 'text-indigo-100' : 'text-indigo-800'}`}>
                        Prediction Results
                      </h3>
                      {alphafoldResult.result.pdb_content && (
                        <button
                          onClick={downloadAlphaFoldPDB}
                          className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-colors ${isDarkMode ? 'bg-indigo-600 hover:bg-indigo-700 text-white' : 'bg-indigo-600 hover:bg-indigo-700 text-white'}`}
                        >
                          <Download className="w-4 h-4" />
                          Download PDB
                        </button>
                      )}
                    </div>
                    {alphafoldResult.result.plddt_score && (
                      <div className={`rounded-lg p-4 border ${isDarkMode ? 'bg-slate-800/50 border-indigo-600' : 'bg-white/50 border-indigo-200'}`}>
                        <h4 className={`text-lg font-semibold mb-2 transition-colors duration-300 ${isDarkMode ? 'text-indigo-100' : 'text-indigo-800'}`}>
                          Confidence Score (pLDDT)
                        </h4>
                        <div className={`text-3xl font-bold mb-2 ${isDarkMode ? 'text-indigo-300' : 'text-indigo-600'}`}>
                          {typeof alphafoldResult.result.plddt_score === 'number' 
                            ? alphafoldResult.result.plddt_score.toFixed(2)
                            : alphafoldResult.result.plddt_score}
                        </div>
                        <p className={`text-sm ${isDarkMode ? 'text-indigo-200' : 'text-indigo-600'}`}>
                          {typeof alphafoldResult.result.plddt_score === 'number' && (
                            <>
                              {alphafoldResult.result.plddt_score >= 90 ? 'Very high confidence' :
                               alphafoldResult.result.plddt_score >= 70 ? 'Confident' :
                               alphafoldResult.result.plddt_score >= 50 ? 'Low confidence' :
                               'Very low confidence'}
                              {' '}(pLDDT: 0-100 scale)
                            </>
                          )}
                        </p>
                      </div>
                    )}
                    {alphafoldResult.result.pdb_content && (
                      <div className={`rounded-lg p-4 border ${isDarkMode ? 'bg-slate-900 border-slate-600' : 'bg-gray-900'}`}>
                        <h4 className={`text-lg font-semibold mb-2 transition-colors duration-300 ${isDarkMode ? 'text-indigo-100' : 'text-indigo-800'}`}>
                          PDB File Content (Preview)
                        </h4>
                        <pre className="text-green-400 text-xs overflow-auto max-h-64 font-mono">
                          {alphafoldResult.result.pdb_content.substring(0, 1000)}
                          {alphafoldResult.result.pdb_content.length > 1000 && '...'}
                        </pre>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Docking Tab Content */}
          {activeTab === 'docking' && (
            <div className="glass-panel rounded-3xl p-8 mb-12 border border-teal-500/20 shadow-2xl space-y-6">
              <div className="space-y-6">
                <div>
                  <label htmlFor="docking-smiles" className="block text-xl font-bold text-white mb-2">
                    SMILES String (Drug Molecule)
                  </label>
                  <input
                    id="docking-smiles"
                    type="text"
                    value={dockingSmiles}
                    onChange={(e) => setDockingSmiles(e.target.value)}
                    placeholder="Enter SMILES string (e.g., C[C@H](N)C(=O)O)..."
                    className="w-full px-6 py-4 rounded-2xl bg-slate-950/80 border border-slate-700 focus:border-teal-400 text-teal-300 placeholder-slate-500 font-mono text-sm outline-none transition-all duration-300 focus:ring-2 focus:ring-teal-500/20"
                    disabled={dockingLoading}
                  />
                </div>
                <div>
                  <label className="flex items-center gap-3 mb-4 cursor-pointer text-slate-300 text-sm font-medium">
                    <input
                      type="checkbox"
                      checked={useProteinSequence}
                      onChange={(e) => setUseProteinSequence(e.target.checked)}
                      className="w-4 h-4 rounded bg-slate-950 border-slate-700 text-teal-500 focus:ring-teal-500"
                    />
                    <span>Use Protein Sequence (otherwise paste PDB file output from AlphaFold2)</span>
                  </label>
                </div>
                {useProteinSequence ? (
                  <div>
                    <label htmlFor="docking-protein-sequence" className="block text-xl font-bold text-white mb-2">
                      Protein Sequence
                    </label>
                    <textarea
                      id="docking-protein-sequence"
                      value={dockingProteinSequence}
                      onChange={(e) => setDockingProteinSequence(e.target.value)}
                      placeholder="Enter protein sequence..."
                      className="w-full h-32 px-6 py-4 rounded-2xl bg-slate-950/80 border border-slate-700 focus:border-teal-400 text-teal-300 placeholder-slate-500 font-mono text-sm leading-relaxed outline-none transition-all"
                      disabled={dockingLoading}
                    />
                  </div>
                ) : (
                  <div>
                    <label htmlFor="docking-protein-pdb" className="block text-xl font-bold text-white mb-2">
                      Protein PDB File Content
                    </label>
                    <textarea
                      id="docking-protein-pdb"
                      value={dockingProteinPDB}
                      onChange={(e) => setDockingProteinPDB(e.target.value)}
                      placeholder="Paste PDB file content here..."
                      className="w-full h-40 px-6 py-4 rounded-2xl bg-slate-950/80 border border-slate-700 focus:border-teal-400 text-teal-300 placeholder-slate-500 font-mono text-sm leading-relaxed outline-none transition-all"
                      disabled={dockingLoading}
                    />
                  </div>
                )}
                <div className="flex flex-col md:flex-row md:items-center md:gap-6 gap-3">
                  <button
                    onClick={handleDocking}
                    disabled={dockingLoading || !dockingSmiles.trim()}
                    className="px-8 py-4 rounded-xl font-bold text-white bg-gradient-to-r from-teal-500 to-cyan-600 hover:from-teal-400 hover:to-cyan-500 shadow-[0_0_20px_rgba(20,184,166,0.4)] transition-all duration-300 flex items-center justify-center gap-3 text-base disabled:opacity-50"
                  >
                    {dockingLoading ? (
                      <>
                        <Loader className="w-5 h-5 animate-spin" />
                        <span>Running AutoDock Vina...</span>
                      </>
                    ) : (
                      <>
                        <FlaskConical className="w-5 h-5" />
                        <span>Run Docking</span>
                      </>
                    )}
                  </button>
                  {dockingError && (
                    <div className="flex items-center gap-3 p-4 bg-red-950/60 border border-red-500/40 rounded-xl text-red-300 text-xs animate-fadeIn">
                      <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
                      <p>{dockingError}</p>
                    </div>
                  )}
                </div>
                {dockingResult && dockingResult.success && (
                  <div className="space-y-6 animate-fadeIn mt-6">
                    <div className="flex items-center gap-3 p-5 bg-green-50 border border-green-200 rounded-xl">
                      <CheckCircle className="w-6 h-6 text-green-500 flex-shrink-0" />
                      <div>
                        <p className="text-green-800 font-semibold text-lg">Docking Completed</p>
                        <p className="text-green-600 text-sm">Binding affinity scores calculated</p>
                      </div>
                    </div>
                    <div className={`rounded-xl p-8 space-y-6 transition-all duration-300 ${isDarkMode ? 'bg-slate-700/50 border border-teal-600' : 'bg-teal-50'}`}>
                      <div className="flex items-center justify-between">
                        <h3 className={`text-2xl font-bold transition-colors duration-300 ${isDarkMode ? 'text-teal-100' : 'text-teal-800'}`}>
                          Docking Results
                        </h3>
                        {dockingResult.result.pdbqt_content && (
                          <button
                            onClick={downloadDockingPDBQT}
                            className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-colors ${isDarkMode ? 'bg-teal-600 hover:bg-teal-700 text-white' : 'bg-teal-600 hover:bg-teal-700 text-white'}`}
                          >
                            <Download className="w-4 h-4" />
                            Download PDBQT
                          </button>
                        )}
                      </div>
                      {dockingResult.result.best_affinity !== undefined && (() => {
                        const score = dockingResult.result.best_affinity;
                        const interp = score <= -9.0 ? {
                          status: "SAFE TO USE DRUG — VERY STRONG BINDING",
                          badgeClass: "bg-emerald-500/20 text-emerald-300 border-emerald-500/50 shadow-[0_0_20px_rgba(16,185,129,0.35)]",
                          textColor: "text-emerald-400",
                          description: "Very strong binding (< -9.0 kcal/mol). High affinity interaction with target protein — safe & highly viable candidate for drug synthesis.",
                          isSafe: true
                        } : score <= -7.5 ? {
                          status: "SAFE & VIABLE LEAD COMPOUND",
                          badgeClass: "bg-cyan-500/20 text-cyan-300 border-cyan-500/50 shadow-[0_0_20px_rgba(6,182,212,0.35)]",
                          textColor: "text-cyan-400",
                          description: "Good to strong binding (-7.5 to -8.9 kcal/mol). Standard threshold for viable lead compounds in drug discovery.",
                          isSafe: true
                        } : score <= -6.0 ? {
                          status: "MODERATE BINDING — REQUIRES OPTIMIZATION",
                          badgeClass: "bg-amber-500/20 text-amber-300 border-amber-500/50 shadow-[0_0_20px_rgba(245,158,11,0.35)]",
                          textColor: "text-amber-400",
                          description: "Moderate binding (-6.0 to -7.4 kcal/mol). Typical starting point for hit optimization in drug discovery.",
                          isSafe: false
                        } : {
                          status: "WEAK BINDING — NOT RECOMMENDED",
                          badgeClass: "bg-red-500/20 text-red-300 border-red-500/50 shadow-[0_0_20px_rgba(239,68,68,0.35)]",
                          textColor: "text-red-400",
                          description: "Weak binding (> -6.0 kcal/mol). The ligand likely does not bind well to the target protein.",
                          isSafe: false
                        };

                        return (
                          <div className="space-y-6">
                            {/* Main Safety Assessment Card */}
                            <div className={`p-6 rounded-3xl border transition-all duration-300 ${interp.badgeClass}`}>
                              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-3">
                                <div className="flex items-center gap-3">
                                  <div className={`p-2.5 rounded-2xl ${interp.isSafe ? 'bg-emerald-500/20 text-emerald-300' : 'bg-amber-500/20 text-amber-300'}`}>
                                    <FlaskConical className="w-7 h-7" />
                                  </div>
                                  <div>
                                    <span className="text-xs uppercase tracking-wider text-slate-400 font-bold block">Drug Viability Assessment</span>
                                    <h4 className={`text-xl font-extrabold tracking-tight ${interp.textColor}`}>{interp.status}</h4>
                                  </div>
                                </div>
                                <div className="text-right">
                                  <span className="text-xs text-slate-400 block font-medium">Best Affinity</span>
                                  <span className={`text-4xl font-black ${interp.textColor}`}>{score.toFixed(2)} <span className="text-sm font-normal text-slate-300">kcal/mol</span></span>
                                </div>
                              </div>
                              <p className="text-slate-200 text-sm font-light leading-relaxed border-t border-white/10 pt-3 mt-2">
                                {interp.description}
                              </p>
                            </div>

                            {/* Docking Score Interpretation Guide */}
                            <div className="glass-panel p-6 rounded-3xl border border-slate-700/60 space-y-4">
                              <h4 className="text-base font-bold text-white flex items-center gap-2">
                                <Sparkles className="w-4 h-4 text-cyan-400" />
                                <span>Docking Score Interpretation Guide (kcal/mol)</span>
                              </h4>
                              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
                                <div className={`p-3.5 rounded-2xl border ${score <= -9.0 ? 'bg-emerald-950/80 border-emerald-500 text-emerald-200 shadow-md' : 'bg-slate-950/60 border-slate-800 text-slate-400'}`}>
                                  <div className="font-bold text-emerald-400 text-sm mb-1">Below -9.0 kcal/mol</div>
                                  <div className="font-semibold text-emerald-300 mb-1">Safe to Use / Very Strong</div>
                                  <p className="text-[11px] leading-tight">High affinity interaction. Highly effective candidate.</p>
                                </div>
                                <div className={`p-3.5 rounded-2xl border ${score > -9.0 && score <= -7.5 ? 'bg-cyan-950/80 border-cyan-500 text-cyan-200 shadow-md' : 'bg-slate-950/60 border-slate-800 text-slate-400'}`}>
                                  <div className="font-bold text-cyan-400 text-sm mb-1">-7.5 to -8.9 kcal/mol</div>
                                  <div className="font-semibold text-cyan-300 mb-1">Safe Lead Compound</div>
                                  <p className="text-[11px] leading-tight">Good to strong binding. Standard viable threshold.</p>
                                </div>
                                <div className={`p-3.5 rounded-2xl border ${score > -7.5 && score <= -6.0 ? 'bg-amber-950/80 border-amber-500 text-amber-200 shadow-md' : 'bg-slate-950/60 border-slate-800 text-slate-400'}`}>
                                  <div className="font-bold text-amber-400 text-sm mb-1">-6.0 to -7.4 kcal/mol</div>
                                  <div className="font-semibold text-amber-300 mb-1">Moderate Binding</div>
                                  <p className="text-[11px] leading-tight">Typical starting point. Optimization required.</p>
                                </div>
                                <div className={`p-3.5 rounded-2xl border ${score > -6.0 ? 'bg-red-950/80 border-red-500 text-red-200 shadow-md' : 'bg-slate-950/60 border-slate-800 text-slate-400'}`}>
                                  <div className="font-bold text-red-400 text-sm mb-1">Above -6.0 kcal/mol</div>
                                  <div className="font-semibold text-red-300 mb-1">Weak Binding</div>
                                  <p className="text-[11px] leading-tight">Ligand does not bind well. Not recommended.</p>
                                </div>
                              </div>
                            </div>
                          </div>
                        );
                      })()}
                      {dockingResult.result.affinities && dockingResult.result.affinities.length > 0 && (
                        <div className="glass-panel p-6 rounded-3xl border border-slate-700/60 space-y-4">
                          <h4 className="text-base font-bold text-white">
                            All Pose Binding Affinities
                          </h4>
                          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                            {dockingResult.result.affinities.map((affinity: number, idx: number) => (
                              <div key={idx} className="text-center p-3.5 rounded-2xl bg-slate-950 border border-slate-800">
                                <div className="text-[11px] text-slate-400 mb-1 font-semibold">Pose {idx + 1}</div>
                                <div className="text-xl font-black text-cyan-300">
                                  {affinity.toFixed(2)}
                                </div>
                                <div className="text-[10px] text-slate-400 font-mono">kcal/mol</div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="mt-12 text-center">
          <p className={`text-lg transition-colors duration-300 ${isDarkMode ? 'text-blue-200' : 'text-blue-200'}`}>
            Powered by advanced machine learning algorithms
          </p>
        </div>
      </div>
    </div>
  );
}

