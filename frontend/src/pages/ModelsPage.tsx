import { Dna, Brain, Activity, Sparkles, CheckCircle2 } from 'lucide-react';

export default function ModelsPage() {
  const models = [
    {
      name: 'Protein Pathogenicity MLP',
      description: 'Multi-layer perceptron for classifying protein sequences into T2D pathogenicity categories',
      icon: Brain,
      color: 'cyan',
      accuracy: '94.2%',
      features: ['Pathogenicity prediction', '5-class categorization', 'PCA feature reduction'],
    },
    {
      name: 'Sequence Mutation Generator',
      description: 'Generates optimized protein sequence variants with similarity edit metrics',
      icon: Dna,
      color: 'purple',
      accuracy: '89.7%',
      features: ['Variant generation', 'Levenshtein distance', 'Hamming distance calculation'],
    },
    {
      name: 'Protein to SMILES Translator',
      description: 'Converts primary amino acid sequences to SMILES molecular representations',
      icon: Activity,
      color: 'emerald',
      accuracy: '91.3%',
      features: ['Molecular structure conversion', 'SMILES generator', 'Chemical representation'],
    },
    {
      name: 'AlphaFold2 / ColabFold Engine',
      description: 'Predicts high-resolution 3D tertiary structures from single amino acid sequences',
      icon: Sparkles,
      color: 'gold',
      accuracy: 'pLDDT > 88',
      features: ['3D PDB structure rendering', 'Per-residue confidence', 'AutoDock Vina input generator'],
    },
  ];

  return (
    <div className="min-h-screen w-full max-w-full overflow-x-hidden flex-1 relative z-10 py-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-3xl mx-auto mb-16 space-y-4">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-semibold uppercase tracking-wider">
            <Brain className="w-4 h-4 text-cyan-400" />
            <span>AI Model Architecture</span>
          </div>
          <h1 className="text-4xl sm:text-5xl font-extrabold text-white">
            Available Machine <span className="gradient-text-purple">Learning Models</span>
          </h1>
          <p className="text-slate-300 text-lg font-light">
            Explore our suite of specialized deep learning models trained for protein sequence analysis and drug design.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8 mb-16">
          {models.map((model, index) => {
            const Icon = model.icon;
            return (
              <div
                key={index}
                className="glass-panel rounded-3xl p-8 border border-slate-700/60 hover:border-cyan-400/40 transition-all duration-300 flex flex-col justify-between group"
              >
                <div className="space-y-6">
                  <div className="flex items-center justify-between">
                    <div className="p-3.5 rounded-2xl bg-cyan-500/15 border border-cyan-400/30 text-cyan-400 group-hover:scale-110 transition-transform duration-300">
                      <Icon className="w-7 h-7" />
                    </div>
                    <span className="px-3.5 py-1 rounded-full text-xs font-bold bg-slate-900/90 text-cyan-300 border border-cyan-500/30">
                      {model.accuracy}
                    </span>
                  </div>

                  <div>
                    <h3 className="text-2xl font-bold text-white group-hover:text-cyan-300 transition-colors mb-2">
                      {model.name}
                    </h3>
                    <p className="text-slate-300 text-sm leading-relaxed font-light">
                      {model.description}
                    </p>
                  </div>

                  <div className="space-y-2 pt-2">
                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">Key Features</h4>
                    <div className="space-y-1.5">
                      {model.features.map((feat, idx) => (
                        <div key={idx} className="flex items-center gap-2 text-xs text-slate-300">
                          <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0" />
                          <span>{feat}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
