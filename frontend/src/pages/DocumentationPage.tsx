import { BookOpen, Code2, Terminal, HelpCircle } from 'lucide-react';

export default function DocumentationPage() {
  return (
    <div className="min-h-screen w-full max-w-full overflow-x-hidden flex-1 relative z-10 py-16">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 w-full">
        <div className="text-center mb-12 space-y-4">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-semibold uppercase tracking-wider">
            <BookOpen className="w-4 h-4 text-cyan-400" />
            <span>Developer & API Guide</span>
          </div>
          <h1 className="text-4xl sm:text-5xl font-extrabold text-white">
            System <span className="gradient-text-cyan">Documentation</span>
          </h1>
          <p className="text-slate-300 text-lg font-light">
            Comprehensive guide to input formats, API endpoints, and platform usage.
          </p>
        </div>

        <div className="glass-panel rounded-3xl p-8 sm:p-10 border border-cyan-500/20 space-y-10 shadow-2xl w-full">
          <section className="space-y-4">
            <h2 className="text-2xl font-bold text-white flex items-center gap-2">
              <Terminal className="w-6 h-6 text-cyan-400" />
              Getting Started
            </h2>
            <p className="text-slate-300 leading-relaxed font-light text-sm">
              Welcome to the Insulin Drug Synthesis platform. Follow these simple steps to perform protein sequence classification, mutation generation, and structural docking.
            </p>
            <div className="bg-slate-950/80 rounded-2xl p-5 border border-slate-800 space-y-3">
              <h3 className="font-semibold text-cyan-300 text-sm">Quick Start Checklist</h3>
              <ol className="list-decimal list-inside space-y-2 text-slate-300 text-xs font-light">
                <li>Create a free account or log in with Guest credentials</li>
                <li>Navigate to the Dashboard Suite</li>
                <li>Enter your target amino acid sequence in single-letter code format</li>
                <li>Select the module tab (Classification, Variant Generation, SMILES, AlphaFold2, or Docking)</li>
                <li>Analyze predictions, view probability distributions, or download PDB structure files</li>
              </ol>
            </div>
          </section>

          <section className="space-y-4 border-t border-slate-800/80 pt-8">
            <h2 className="text-2xl font-bold text-white flex items-center gap-2">
              <Code2 className="w-6 h-6 text-purple-400" />
              Backend API Endpoints
            </h2>
            <div className="space-y-3">
              <div className="bg-slate-950/80 rounded-xl p-4 border border-slate-800 flex items-start gap-3">
                <span className="px-2.5 py-1 rounded-md text-xs font-mono font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                  POST /predict
                </span>
                <p className="text-slate-300 text-xs pt-0.5">Classify protein sequence pathogenicity</p>
              </div>

              <div className="bg-slate-950/80 rounded-xl p-4 border border-slate-800 flex items-start gap-3">
                <span className="px-2.5 py-1 rounded-md text-xs font-mono font-bold bg-purple-500/20 text-purple-300 border border-purple-500/30">
                  POST /generate-sequences
                </span>
                <p className="text-slate-300 text-xs pt-0.5">Generate novel protein sequence variants with edit metrics</p>
              </div>

              <div className="bg-slate-950/80 rounded-xl p-4 border border-slate-800 flex items-start gap-3">
                <span className="px-2.5 py-1 rounded-md text-xs font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                  POST /generate-smiles
                </span>
                <p className="text-slate-300 text-xs pt-0.5">Convert amino acid sequence to SMILES chemical structure</p>
              </div>
            </div>
          </section>

          <section className="space-y-4 border-t border-slate-800/80 pt-8">
            <h2 className="text-2xl font-bold text-white flex items-center gap-2">
              <Terminal className="w-6 h-6 text-emerald-400" />
              Sequence Format Standard
            </h2>
            <p className="text-slate-300 leading-relaxed font-light text-sm">
              Protein sequences must be provided in standard single-letter amino acid code format. Valid amino acid single-letter codes include: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y.
            </p>
            <div className="bg-slate-950 rounded-2xl p-5 border border-emerald-500/30 max-w-full overflow-x-auto shadow-inner">
              <div className="text-[10px] uppercase font-mono text-emerald-400/80 mb-2">Example Sequence Input</div>
              <code className="text-emerald-400 text-xs font-mono break-all whitespace-pre-wrap block max-w-full leading-relaxed">
                MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWQTSTSTSLPRADLQLFVDGVRQLEWLSQRLQQPQQKSAFAVQEDFNRSWFRPGHRRNKVFDLPIGVLKSSAQNLMNQEDVHSKQAPGTILKSQGMQVFVLEELDKTLFTLGFHKPAIVQHASSAKDLGPLLDGIWKTTTTKQAAKCLQKNLPSFLGVTSSEFRYLMNSQTRLPDNYLPLLPAIIDRFDNTLPLTGQAQIIFRRFLPLQGKEFQ
              </code>
            </div>
          </section>

          <section className="space-y-4 border-t border-slate-800/80 pt-8">
            <h2 className="text-2xl font-bold text-white flex items-center gap-2">
              <HelpCircle className="w-6 h-6 text-indigo-400" />
              Frequently Asked Questions
            </h2>
            <div className="space-y-4 text-xs font-light text-slate-300">
              <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800 space-y-1">
                <h3 className="font-bold text-white text-sm">What is the minimum recommended sequence length?</h3>
                <p>Sequences should be at least 10 amino acids long for optimal model feature extraction.</p>
              </div>

              <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800 space-y-1">
                <h3 className="font-bold text-white text-sm">Are sequences permanently stored in a backend database?</h3>
                <p>No. Sequences are processed dynamically in memory and are not stored permanently.</p>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
