import { Shield, Cpu, Dna, Sparkles, Terminal } from 'lucide-react';

export default function AboutPage() {
  return (
    <div className="min-h-screen w-full max-w-full overflow-x-hidden flex-1 relative z-10 py-16">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12 space-y-4">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-semibold uppercase tracking-wider">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            <span>Platform Overview</span>
          </div>
          <h1 className="text-4xl sm:text-5xl font-extrabold text-white">
            About <span className="gradient-text-cyan">Insulin Drug Synthesis</span>
          </h1>
          <p className="text-slate-300 text-lg max-w-2xl mx-auto font-light">
            Merging machine learning intelligence with structural biology for accelerated diabetes drug research.
          </p>
        </div>

        <div className="glass-panel rounded-3xl p-8 sm:p-10 border border-cyan-500/20 space-y-10 shadow-2xl">
          <section className="space-y-3">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-cyan-500/20 border border-cyan-500/40 text-cyan-400">
                <Dna className="w-6 h-6" />
              </div>
              <h2 className="text-2xl font-bold text-white">Our Mission</h2>
            </div>
            <p className="text-slate-300 leading-relaxed font-light">
              Insulin Drug Synthesis is an AI-native platform designed to revolutionize the process of diabetes drug discovery and protein variant analysis. We combine custom deep learning architectures with intuitive Web interfaces to make complex biochemical analysis instant and seamless.
            </p>
          </section>

          <section className="space-y-4 border-t border-slate-800/80 pt-8">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-purple-500/20 border border-purple-500/40 text-purple-400">
                <Cpu className="w-6 h-6" />
              </div>
              <h2 className="text-2xl font-bold text-white">Core Capabilities</h2>
            </div>
            <div className="grid sm:grid-cols-3 gap-4 pt-2">
              <div className="glass-card p-5 rounded-2xl border border-cyan-500/20 space-y-2">
                <div className="text-cyan-400 font-bold text-base">01. Classification</div>
                <p className="text-slate-300 text-xs leading-relaxed">
                  Pathogenicity prediction using trained MLP classifiers and PCA dimensionality reduction.
                </p>
              </div>

              <div className="glass-card-purple p-5 rounded-2xl border border-purple-500/20 space-y-2">
                <div className="text-purple-400 font-bold text-base">02. Sequence Gen</div>
                <p className="text-slate-300 text-xs leading-relaxed">
                  Generate novel sequence variants with Levenshtein and Hamming edit similarity metrics.
                </p>
              </div>

              <div className="glass-card-emerald p-5 rounded-2xl border border-emerald-500/20 space-y-2">
                <div className="text-emerald-400 font-bold text-base">03. SMILES & Docking</div>
                <p className="text-slate-300 text-xs leading-relaxed">
                  Convert sequences into molecular structures & dock them with protein receptors using AutoDock Vina.
                </p>
              </div>
            </div>
          </section>

          <section className="space-y-3 border-t border-slate-800/80 pt-8">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-emerald-500/20 border border-emerald-500/40 text-emerald-400">
                <Terminal className="w-6 h-6" />
              </div>
              <h2 className="text-2xl font-bold text-white">Technology Stack</h2>
            </div>
            <p className="text-slate-300 leading-relaxed font-light">
              Our backend leverages PyTorch, ColabFold (AlphaFold2), AutoDock Vina, and Flask API microservices. The frontend interface is built with React, TypeScript, Vite, and Tailwind CSS with custom WebGL/Canvas graphics.
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
