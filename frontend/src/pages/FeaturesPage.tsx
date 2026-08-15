import { Activity, Dna, FlaskConical, Brain, Zap, ShieldCheck, Sparkles, CheckCircle2 } from 'lucide-react';

export default function FeaturesPage() {
  const features = [
    {
      icon: Activity,
      title: 'Real-time Sequence Classification',
      description: 'Get instant predictions and classifications for your protein sequences with confidence scores.',
    },
    {
      icon: Dna,
      title: 'Variant Sequence Generation',
      description: 'Generate mutant protein variants ranked by Levenshtein and Hamming edit distance similarity.',
    },
    {
      icon: FlaskConical,
      title: 'SMILES Molecular Structure Conversion',
      description: 'Convert protein sequences into SMILES representations for ligand design and chemical simulation.',
    },
    {
      icon: Brain,
      title: 'AlphaFold2 3D Folding',
      description: 'Leverage ColabFold GPU backend to fold amino acid chains into 3D PDB structure representations.',
    },
    {
      icon: Zap,
      title: 'AutoDock Vina Docking',
      description: 'Calculate molecular binding affinities (kcal/mol) between drug candidates and receptor targets.',
    },
    {
      icon: ShieldCheck,
      title: 'Secure & Private Platform',
      description: 'In-memory processing ensures sequence input privacy and zero permanent storage leaks.',
    },
  ];

  return (
    <div className="min-h-screen w-full max-w-full overflow-x-hidden flex-1 relative z-10 py-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-3xl mx-auto mb-16 space-y-4">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-semibold uppercase tracking-wider">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            <span>Platform Features</span>
          </div>
          <h1 className="text-4xl sm:text-5xl font-extrabold text-white">
            Designed for <span className="gradient-text-cyan">Biomedical Breakthroughs</span>
          </h1>
          <p className="text-slate-300 text-lg font-light">
            Comprehensive computational toolsets tailored for protein engineering and diabetes drug discovery.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <div
                key={index}
                className="glass-panel rounded-3xl p-7 border border-slate-700/60 hover:border-cyan-400/40 transition-all duration-300 space-y-4 group"
              >
                <div className="w-14 h-14 rounded-2xl bg-cyan-500/15 border border-cyan-400/30 flex items-center justify-center text-cyan-400 group-hover:scale-110 transition-transform duration-300">
                  <Icon className="w-7 h-7" />
                </div>
                <h3 className="text-xl font-bold text-white group-hover:text-cyan-300 transition-colors">
                  {feature.title}
                </h3>
                <p className="text-slate-300 text-sm leading-relaxed font-light">
                  {feature.description}
                </p>
              </div>
            );
          })}
        </div>

        {/* Workflow Section */}
        <div className="glass-panel rounded-3xl p-8 sm:p-10 border border-cyan-500/20 space-y-8 shadow-2xl">
          <h2 className="text-3xl font-extrabold text-white text-center">How The Discovery Suite Works</h2>

          <div className="grid sm:grid-cols-3 gap-8 pt-4">
            <div className="space-y-3 text-center sm:text-left">
              <div className="w-10 h-10 rounded-full bg-cyan-500 text-slate-950 font-black flex items-center justify-center text-lg mx-auto sm:mx-0 shadow-[0_0_15px_rgba(6,182,212,0.5)]">
                1
              </div>
              <h3 className="text-lg font-bold text-white">Input Sequence</h3>
              <p className="text-slate-300 text-xs font-light leading-relaxed">
                Paste your single-letter amino acid code sequence into the dashboard interface.
              </p>
            </div>

            <div className="space-y-3 text-center sm:text-left">
              <div className="w-10 h-10 rounded-full bg-purple-500 text-slate-950 font-black flex items-center justify-center text-lg mx-auto sm:mx-0 shadow-[0_0_15px_rgba(168,85,247,0.5)]">
                2
              </div>
              <h3 className="text-lg font-bold text-white">Select Analysis Tool</h3>
              <p className="text-slate-300 text-xs font-light leading-relaxed">
                Choose between classification, mutation generation, SMILES conversion, AlphaFold2, or AutoDock Vina.
              </p>
            </div>

            <div className="space-y-3 text-center sm:text-left">
              <div className="w-10 h-10 rounded-full bg-emerald-500 text-slate-950 font-black flex items-center justify-center text-lg mx-auto sm:mx-0 shadow-[0_0_15px_rgba(52,211,153,0.5)]">
                3
              </div>
              <h3 className="text-lg font-bold text-white">Interactive Insights</h3>
              <p className="text-slate-300 text-xs font-light leading-relaxed">
                Download structural PDBQT/PDB files, examine affinity metrics, and view real-time confidence scores.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
