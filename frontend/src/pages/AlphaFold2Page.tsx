import { useState } from 'react';
import { Loader, CheckCircle2, AlertCircle, Dna, Download, Sparkles } from 'lucide-react';

interface AlphaFold2Result {
  success: boolean;
  sequence: string;
  result: {
    pdb_content?: string;
    pdb_url?: string;
    confidence_scores?: Record<string, number>;
    plddt_score?: number;
    error?: string;
  };
  error?: string;
}

export default function AlphaFold2Page() {
  const [sequence, setSequence] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AlphaFold2Result | null>(null);
  const [error, setError] = useState<string | null>(null);

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
      setError('Invalid amino acid sequence. Please use single-letter amino acid codes.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:5001/alphafold2/predict', {
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

      const data: AlphaFold2Result = await response.json();

      if (data.success) {
        setResult(data);
      } else {
        setError(data.error || 'AlphaFold2 prediction failed');
      }
    } catch (err) {
      console.error('AlphaFold2 error:', err);
      setError('Unable to connect to server. Please ensure the backend and Colab notebook are running.');
    } finally {
      setLoading(false);
    }
  };

  const downloadPDB = () => {
    if (result?.result?.pdb_content) {
      const blob = new Blob([result.result.pdb_content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `alphafold2_prediction_${result.sequence.substring(0, 10)}.pdb`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  return (
    <div className="min-h-screen w-full max-w-full overflow-x-hidden flex-1 relative z-10 py-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-3xl mx-auto mb-12 space-y-4">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-semibold uppercase tracking-wider">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            <span>ColabFold GPU Engine</span>
          </div>
          <h1 className="text-4xl sm:text-5xl font-extrabold text-white">
            AlphaFold2 <span className="gradient-text-cyan">Structure Prediction</span>
          </h1>
          <p className="text-slate-300 text-lg font-light">
            Predict 3D tertiary protein structures powered by AlphaFold2 deep neural networks.
          </p>
        </div>

        <div className="glass-panel rounded-3xl p-8 border border-cyan-500/20 shadow-2xl mb-8 space-y-6">
          <label htmlFor="alphafold-sequence" className="block text-xl font-bold text-white">
            Protein Sequence Input
          </label>
          <textarea
            id="alphafold-sequence"
            value={sequence}
            onChange={(e) => setSequence(e.target.value)}
            placeholder="Enter amino acid sequence (e.g., GIVEQCCTSICSLYQLENYCNFVNQHLCGSHLVEALYLVC)..."
            className="w-full h-40 px-6 py-4 rounded-2xl border bg-slate-950/80 border-slate-700 focus:border-cyan-400 text-cyan-300 placeholder-slate-500 font-mono text-sm leading-relaxed outline-none transition-all duration-300 focus:ring-2 focus:ring-cyan-500/20"
            disabled={loading}
          />
          <div className="flex flex-col sm:flex-row items-center gap-4">
            <button
              onClick={handlePredict}
              disabled={loading || !sequence.trim()}
              className="btn-neon-cyan px-8 py-4 rounded-xl font-bold text-white flex items-center justify-center gap-3 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed text-base"
            >
              {loading ? (
                <>
                  <Loader className="w-5 h-5 animate-spin text-cyan-200" />
                  <span>Folding 3D Structure...</span>
                </>
              ) : (
                <>
                  <Dna className="w-5 h-5" />
                  <span>Predict 3D Structure</span>
                </>
              )}
            </button>

            {error && (
              <div className="flex items-center gap-3 p-4 bg-red-950/50 border border-red-500/30 rounded-xl text-red-300 text-sm animate-fadeIn">
                <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
                <p>{error}</p>
              </div>
            )}
          </div>
        </div>

        {result && result.success && (
          <div className="space-y-6 animate-fadeIn">
            <div className="flex items-center gap-3 p-5 bg-emerald-950/50 border border-emerald-500/30 rounded-2xl">
              <CheckCircle2 className="w-6 h-6 text-emerald-400 shrink-0" />
              <div>
                <p className="text-emerald-200 font-bold text-base">3D Structure Folded Successfully</p>
                <p className="text-emerald-400 text-xs">PDB coordinate file generated with confidence score metrics</p>
              </div>
            </div>

            <div className="glass-panel rounded-3xl p-8 border border-slate-700/60 space-y-6 shadow-2xl">
              <div className="flex items-center justify-between">
                <h3 className="text-2xl font-bold text-white">Prediction Output</h3>
                {result.result.pdb_content && (
                  <button
                    onClick={downloadPDB}
                    className="btn-neon-cyan px-5 py-2.5 rounded-xl text-sm font-semibold flex items-center gap-2"
                  >
                    <Download className="w-4 h-4" />
                    <span>Download PDB File</span>
                  </button>
                )}
              </div>

              {result.result.plddt_score && (
                <div className="glass-card p-6 rounded-2xl border border-cyan-500/20">
                  <h4 className="text-sm font-semibold text-slate-300 mb-1">Overall Confidence (pLDDT)</h4>
                  <div className="text-4xl font-extrabold gradient-text-cyan">{result.result.plddt_score.toFixed(2)}</div>
                  <p className="text-slate-400 text-xs mt-2">Scores above 70 indicate high topological confidence.</p>
                </div>
              )}

              {result.result.pdb_content && (
                <div className="bg-slate-950 rounded-2xl p-5 border border-slate-800 space-y-2">
                  <h4 className="text-sm font-bold text-white">PDB File Preview</h4>
                  <pre className="text-emerald-400 text-xs font-mono break-all whitespace-pre-wrap max-h-64 overflow-y-auto leading-relaxed">
                    {result.result.pdb_content.substring(0, 1000)}
                    {result.result.pdb_content.length > 1000 && '...'}
                  </pre>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
