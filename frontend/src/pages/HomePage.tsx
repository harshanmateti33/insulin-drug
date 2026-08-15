import { useNavigate } from 'react-router-dom';
import { Activity, Brain, Dna, FlaskConical, Sparkles, ArrowRight, ShieldCheck, Zap, Layers, Code2, Users } from 'lucide-react';

export default function HomePage() {
  const navigate = useNavigate();

  const teamMembers = [
    { name: 'Devika', category: 'Mentor', roleBg: 'from-amber-500/20 to-orange-500/20', border: 'border-amber-400/40', text: 'text-amber-300' },
    { name: 'Mateti Harshan', category: 'Technical Lead', roleBg: 'from-cyan-500/20 to-blue-500/20', border: 'border-cyan-400/40', text: 'text-cyan-300' },
    { name: 'Mateti Hrushikesh', category: 'AI Engineer', roleBg: 'from-purple-500/20 to-pink-500/20', border: 'border-purple-400/40', text: 'text-purple-300' },
    { name: 'Mallela Nikhil Reddy', category: 'Model Architect', roleBg: 'from-indigo-500/20 to-cyan-500/20', border: 'border-indigo-400/40', text: 'text-indigo-300' },
    { name: 'M.Nithilesh', category: 'Team Leader', roleBg: 'from-emerald-500/20 to-teal-500/20', border: 'border-emerald-400/40', text: 'text-emerald-300' },
    { name: 'M.Jathin', category: 'MERN stack Developer', roleBg: 'from-blue-500/20 to-indigo-500/20', border: 'border-blue-400/40', text: 'text-blue-300' },
  ];

  return (
    <div className="min-h-screen w-full max-w-full overflow-x-hidden flex-1 relative z-10 pb-20">
      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-16 pb-20">
        <div className="grid md:grid-cols-12 gap-12 items-center">
          {/* Left Side */}
          <div className="md:col-span-7 space-y-6">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card border border-cyan-400/30 text-cyan-300 text-xs font-semibold uppercase tracking-wider shadow-[0_0_15px_rgba(6,182,212,0.2)]">
              <Sparkles className="w-4 h-4 text-cyan-400 animate-spin" />
              <span>Next-Gen Computational Biology</span>
            </div>

            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight text-white leading-tight">
              Insulin Drug <br />
              <span className="gradient-text-cyan">Synthesis AI</span>
            </h1>

            <p className="text-slate-300 text-lg sm:text-xl leading-relaxed max-w-2xl font-light">
              Where researchers come to innovate. Analyze protein sequences, generate novel variants, and predict 3D molecular structures—all powered by state-of-the-art deep learning architectures.
            </p>

            <div className="flex flex-wrap gap-4 pt-4">
              <button
                onClick={() => navigate('/dashboard')}
                className="btn-neon-lime px-8 py-4 rounded-xl text-lg font-bold flex items-center gap-3 transform hover:scale-105 transition-all shadow-[0_0_25px_rgba(163,230,53,0.4)]"
              >
                <span>Launch Discovery Suite</span>
                <ArrowRight className="w-5 h-5" />
              </button>

              <button
                onClick={() => navigate('/models')}
                className="px-7 py-4 rounded-xl font-semibold text-slate-200 glass-panel border border-slate-700/60 hover:border-cyan-400/50 hover:text-cyan-300 transition-all duration-300 flex items-center gap-2"
              >
                <Brain className="w-5 h-5 text-purple-400" />
                <span>Explore ML Models</span>
              </button>
            </div>

            {/* Quick Metrics Bar */}
            <div className="grid grid-cols-3 gap-4 pt-8 border-t border-slate-800/80">
              <div className="glass-card p-4 rounded-xl text-center">
                <div className="text-2xl font-bold gradient-text-cyan">94.2%</div>
                <div className="text-xs text-slate-400 mt-1">Pathogenicity Accuracy</div>
              </div>
              <div className="glass-card p-4 rounded-xl text-center">
                <div className="text-2xl font-bold gradient-text-purple">Real-Time</div>
                <div className="text-xs text-slate-400 mt-1">SMILES & Variant Gen</div>
              </div>
              <div className="glass-card p-4 rounded-xl text-center">
                <div className="text-2xl font-bold gradient-text-emerald">AlphaFold2</div>
                <div className="text-xs text-slate-400 mt-1">3D Structure Folding</div>
              </div>
            </div>
          </div>

          {/* Right Side - Interactive Hero Card preview */}
          <div className="md:col-span-5">
            <div className="relative">
              {/* Background ambient glow circle */}
              <div className="absolute -inset-1 rounded-3xl bg-gradient-to-r from-cyan-500 to-purple-600 opacity-30 blur-2xl animate-pulse-glow"></div>

              <div className="relative glass-panel rounded-3xl p-6 border border-cyan-500/30 space-y-6 shadow-2xl">
                <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
                  <div className="flex items-center gap-3">
                    <div className="p-2.5 rounded-xl bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
                      <Dna className="w-6 h-6 animate-pulse" />
                    </div>
                    <div>
                      <h3 className="font-bold text-white text-base">Live Sequence Analyzer</h3>
                      <p className="text-xs text-slate-400">Insulin Receptor Variant B</p>
                    </div>
                  </div>
                  <span className="px-2.5 py-1 rounded-full text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 uppercase tracking-wider">
                    Online
                  </span>
                </div>

                {/* Sample Sequence Box */}
                <div className="bg-slate-950/80 rounded-xl p-4 border border-slate-800 font-mono text-xs text-cyan-300 space-y-2">
                  <div className="flex justify-between text-[10px] text-slate-400 font-sans">
                    <span>INPUT SEQUENCE</span>
                    <span>LENGTH: 42 AA</span>
                  </div>
                  <div className="break-all tracking-wider text-emerald-400 bg-emerald-950/30 p-2 rounded border border-emerald-500/20">
                    GIVEQCCTSICSLYQLENYCNFVNQHLCGSHLVEALYLVC
                  </div>
                </div>

                {/* Live Confidence Metric Bars */}
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-xs text-slate-300 mb-1">
                      <span>Benign Variant Match</span>
                      <span className="font-bold text-cyan-400">96.8%</span>
                    </div>
                    <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full" style={{ width: '96.8%' }}></div>
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs text-slate-300 mb-1">
                      <span>SMILES Conversion Affinity</span>
                      <span className="font-bold text-purple-400">91.4%</span>
                    </div>
                    <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full" style={{ width: '91.4%' }}></div>
                    </div>
                  </div>
                </div>

                {/* Action status pill */}
                <div className="p-3 rounded-xl bg-slate-900/90 border border-slate-800 flex items-center justify-between text-xs text-slate-300">
                  <div className="flex items-center gap-2">
                    <ShieldCheck className="w-4 h-4 text-emerald-400" />
                    <span>ML Pipeline Operational</span>
                  </div>
                  <span className="text-cyan-400 font-mono">0.042s</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Feature Showcase Grid */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center max-w-3xl mx-auto mb-16 space-y-4">
          <h2 className="text-3xl sm:text-4xl font-extrabold text-white">
            Integrated Computational Suite
          </h2>
          <p className="text-slate-400 text-base sm:text-lg">
            Empowering biochemical research with modern machine learning tools for sequence classification, mutation generation, and structural docking.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {/* Card 1: Sequence Analysis */}
          <div className="glass-card rounded-2xl p-6 flex flex-col justify-between group">
            <div className="space-y-4">
              <div className="w-14 h-14 rounded-2xl bg-cyan-500/15 border border-cyan-400/30 flex items-center justify-center text-cyan-400 group-hover:scale-110 transition-transform duration-300">
                <Activity className="w-7 h-7" />
              </div>
              <h3 className="text-2xl font-bold text-white group-hover:text-cyan-300 transition-colors">
                Sequence Classification
              </h3>
              <p className="text-slate-300 text-sm leading-relaxed">
                Multi-layer perceptron (MLP) trained on protein embeddings to classify amino acid sequences into pathogenicity categories.
              </p>
            </div>

            <div className="mt-8 pt-4 border-t border-slate-800/80 space-y-3">
              <div className="flex justify-between items-center text-xs text-slate-400">
                <span>CLASSIFICATION CONFIDENCE</span>
                <span className="text-cyan-400 font-bold">94.2%</span>
              </div>
              <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-cyan-500 to-indigo-500 rounded-full" style={{ width: '94.2%' }}></div>
              </div>
            </div>
          </div>

          {/* Card 2: Model Library */}
          <div className="glass-card-purple rounded-2xl p-6 flex flex-col justify-between group">
            <div className="space-y-4">
              <div className="w-14 h-14 rounded-2xl bg-purple-500/15 border border-purple-400/30 flex items-center justify-center text-purple-400 group-hover:scale-110 transition-transform duration-300">
                <Brain className="w-7 h-7" />
              </div>
              <h3 className="text-2xl font-bold text-white group-hover:text-purple-300 transition-colors">
                Model Library
              </h3>
              <p className="text-slate-300 text-sm leading-relaxed">
                Access pre-trained language models, PCA feature extractors, and protein encoders built for structural biology tasks.
              </p>
            </div>

            <div className="mt-8 pt-4 border-t border-slate-800/80 space-y-3">
              <div className="flex justify-between items-center text-xs text-slate-400">
                <span>ENCODER PERFORMANCE</span>
                <span className="text-purple-400 font-bold">89.7%</span>
              </div>
              <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full" style={{ width: '89.7%' }}></div>
              </div>
            </div>
          </div>

          {/* Card 3: Sequence Generation */}
          <div className="glass-card-emerald rounded-2xl p-6 flex flex-col justify-between group">
            <div className="space-y-4">
              <div className="w-14 h-14 rounded-2xl bg-emerald-500/15 border border-emerald-400/30 flex items-center justify-center text-emerald-400 group-hover:scale-110 transition-transform duration-300">
                <FlaskConical className="w-7 h-7" />
              </div>
              <h3 className="text-2xl font-bold text-white group-hover:text-emerald-300 transition-colors">
                Variant Generation
              </h3>
              <p className="text-slate-300 text-sm leading-relaxed">
                Generate high-affinity sequence mutations ranked by Levenshtein and Hamming edit distance similarity scores.
              </p>
            </div>

            <div className="mt-8 pt-4 border-t border-slate-800/80 space-y-3">
              <div className="flex justify-between items-center text-xs text-slate-400">
                <span>SIMILARITY SCORE</span>
                <span className="text-emerald-400 font-bold">91.3%</span>
              </div>
              <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-emerald-500 to-teal-400 rounded-full" style={{ width: '91.3%' }}></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Team Members Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 border-t border-slate-800/80">
        <div className="text-center max-w-3xl mx-auto mb-16 space-y-3">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-300 text-xs font-semibold uppercase tracking-wider">
            <Users className="w-3.5 h-3.5" />
            <span>Research & Engineering</span>
          </div>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-white">
            TEAM MEMBERS
          </h2>
          <p className="text-slate-400 text-base">
            The dedicated researchers and engineers behind the Insulin Drug Synthesis project.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {teamMembers.map((member, index) => (
            <div
              key={index}
              className="glass-card rounded-2xl p-6 border border-slate-800 hover:border-cyan-500/40 transition-all duration-300 flex items-center gap-4 group"
            >
              <div className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${member.roleBg} border ${member.border} flex items-center justify-center shrink-0 group-hover:scale-110 transition-transform duration-300 shadow-lg`}>
                <span className={`text-2xl font-black ${member.text}`}>
                  {member.name.charAt(0)}
                </span>
              </div>

              <div className="flex-1 min-w-0">
                <h4 className="text-lg font-bold text-white group-hover:text-cyan-300 transition-colors truncate">
                  {member.name}
                </h4>
                <div className={`inline-block px-2.5 py-0.5 mt-1 rounded-md text-xs font-medium bg-slate-900/80 border ${member.border} ${member.text}`}>
                  {member.category}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
