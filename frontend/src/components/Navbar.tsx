import { Link, useLocation } from 'react-router-dom';
import { Activity, Dna, Sparkles } from 'lucide-react';

interface NavbarProps {
  user: { id: string; username: string } | null;
  onLoginClick: () => void;
}

export default function Navbar({ user, onLoginClick }: NavbarProps) {
  const location = useLocation();

  const navLinks = [
    { path: '/models', label: 'Models' },
    { path: '/about', label: 'About' },
    { path: '/features', label: 'Features' },
    { path: '/documentation', label: 'Documentation' },
  ];

  return (
    <nav className="sticky top-0 z-50 w-full max-w-full overflow-x-hidden glass-panel border-b border-cyan-500/20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3 group">
            <div className="relative p-2.5 rounded-xl bg-gradient-to-br from-cyan-500/20 to-purple-500/20 border border-cyan-400/30 group-hover:border-cyan-400/60 transition-all duration-300 shadow-[0_0_15px_rgba(6,182,212,0.25)]">
              <Dna className="w-6 h-6 text-cyan-400 group-hover:rotate-12 transition-transform duration-300" />
              <span className="absolute -top-1 -right-1 flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-cyan-400"></span>
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-xl font-extrabold tracking-tight text-white flex items-center gap-1.5">
                Insulin <span className="gradient-text-cyan">Drug Synthesis</span>
              </span>
              <span className="text-[10px] uppercase tracking-widest text-cyan-400/80 font-mono flex items-center gap-1">
                <Sparkles className="w-3 h-3 text-cyan-400 animate-pulse" /> AI Discovery Platform
              </span>
            </div>
          </Link>

          {/* Navigation Links */}
          <div className="hidden md:flex items-center gap-1 bg-slate-900/60 p-1.5 rounded-full border border-slate-700/50 backdrop-blur-md">
            {navLinks.map((link) => {
              const isActive = location.pathname === link.path;
              return (
                <Link
                  key={link.path}
                  to={link.path}
                  className={`px-5 py-2 rounded-full text-sm font-medium transition-all duration-300 relative ${
                    isActive
                      ? 'text-cyan-300 font-semibold bg-cyan-500/15 border border-cyan-500/30 shadow-[0_0_12px_rgba(6,182,212,0.2)]'
                      : 'text-slate-300 hover:text-cyan-300 hover:bg-slate-800/50'
                  }`}
                >
                  {link.label}
                </Link>
              );
            })}
          </div>

          {/* Right Side Actions */}
          <div className="flex items-center gap-3">
            {user ? (
              <Link
                to="/dashboard"
                className="px-5 py-2.5 bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-white rounded-xl text-sm font-semibold shadow-[0_0_20px_rgba(6,182,212,0.4)] transition-all duration-300 flex items-center gap-2 transform hover:scale-105"
              >
                <Activity className="w-4 h-4 text-cyan-200" />
                Dashboard
              </Link>
            ) : (
              <>
                <button
                  onClick={onLoginClick}
                  className="px-5 py-2.5 bg-slate-800/80 hover:bg-slate-700/80 border border-slate-600/50 text-slate-200 rounded-xl text-sm font-medium transition-all duration-300 hidden sm:block hover:border-cyan-400/40"
                >
                  Login
                </button>
                <button
                  onClick={onLoginClick}
                  className="btn-neon-lime px-6 py-2.5 rounded-xl text-sm font-bold transition-all duration-300 flex items-center gap-1.5"
                >
                  Sign up
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
