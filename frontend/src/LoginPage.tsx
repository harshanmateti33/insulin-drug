import React, { useState } from 'react';
import { Eye, EyeOff, User, Lock, LogIn, UserPlus, AlertCircle, CheckCircle2, Sparkles, Smartphone, Mail, Key, ArrowLeft, Home } from 'lucide-react';
import { API_BASE_URL } from './config';

interface LoginPageProps {
  onLogin: (user: { id: string; username: string }) => void;
  onBackToHome?: () => void;
}

interface AuthResponse {
  success: boolean;
  message: string;
  user?: {
    id: string;
    username: string;
  };
  error?: string;
}

const LoginPage: React.FC<LoginPageProps> = ({ onLogin, onBackToHome }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [selectedAuthMethod, setSelectedAuthMethod] = useState<'default' | 'google' | 'mobile'>('default');

  // Email OTP flow
  const [email, setEmail] = useState('');
  const [emailOtp, setEmailOtp] = useState('');
  const [emailStep, setEmailStep] = useState<'collect' | 'otp' | 'setPassword'>('collect');

  // Mobile OTP flow
  const [mobile, setMobile] = useState('');
  const [mobileOtp, setMobileOtp] = useState('');
  const [mobileStep, setMobileStep] = useState<'collect' | 'otp' | 'setPassword'>('collect');

  // New password for OTP signup
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  // Password strength helper
  const passwordStrength = (pwd: string) => {
    let score = 0;
    if (pwd.length >= 8) score++;
    if (/[A-Z]/.test(pwd)) score++;
    if (/[a-z]/.test(pwd)) score++;
    if (/\d/.test(pwd)) score++;
    if (/[^A-Za-z0-9]/.test(pwd)) score++;
    const percent = Math.min(100, (score / 5) * 100);
    const label = score <= 2 ? 'Weak' : score === 3 ? 'Fair' : score === 4 ? 'Good' : 'Strong';
    const color = score <= 2 ? 'bg-red-500' : score === 3 ? 'bg-amber-500' : score === 4 ? 'bg-emerald-500' : 'bg-cyan-400';
    return { score, percent, label, color };
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const endpoint = isLogin ? '/api/login' : '/api/register';
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ username, password }),
      });

      const data: AuthResponse = await response.json();

      if (data.success) {
        setSuccess(data.message);
        if (isLogin && data.user) {
          setTimeout(() => {
            onLogin(data.user!);
          }, 800);
        } else if (!isLogin) {
          setTimeout(() => {
            setIsLogin(true);
            setSuccess(null);
          }, 1500);
        }
      } else {
        setError(data.error || 'An error occurred during authentication');
      }
    } catch (err) {
      console.error('Auth error:', err);
      setError('Unable to connect to server. Please verify backend is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const toggleMode = () => {
    setIsLogin(!isLogin);
    setError(null);
    setSuccess(null);
    setUsername('');
    setPassword('');
    setSelectedAuthMethod('default');
    setEmail('');
    setEmailOtp('');
    setEmailStep('collect');
    setMobile('');
    setMobileOtp('');
    setMobileStep('collect');
    setNewPassword('');
    setConfirmPassword('');
  };

  const handleSelectAuth = (method: 'default' | 'google' | 'mobile') => {
    const next = selectedAuthMethod === method ? 'default' : method;
    setSelectedAuthMethod(next);
    setError(null);
    setSuccess(null);
    setEmail('');
    setEmailOtp('');
    setEmailStep('collect');
    setMobile('');
    setMobileOtp('');
    setMobileStep('collect');
    setNewPassword('');
    setConfirmPassword('');
  };

  const handleGoogleAuth = () => {
    setIsLoading(true);
    setError(null);
    window.location.href = `${API_BASE_URL}/api/auth/google`;
  };

  const requestMobileOtp = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const res = await fetch(`${API_BASE_URL}/api/request-otp-mobile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ mobile, username }),
      });
      const data = await res.json();
      if (data.success) {
        setSuccess('OTP sent to mobile');
        setMobileStep('otp');
      } else {
        setError(data.error || 'Failed to send OTP');
      }
    } catch (e) {
      setError('Unable to request OTP');
    } finally {
      setIsLoading(false);
    }
  };

  const verifyMobileOtp = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const res = await fetch(`${API_BASE_URL}/api/verify-otp-mobile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ mobile, otp: mobileOtp }),
      });
      const data = await res.json();
      if (data.success) {
        setSuccess('OTP verified');
        setMobileStep('setPassword');
      } else {
        setError(data.error || 'Invalid OTP');
      }
    } catch (e) {
      setError('Unable to verify OTP');
    } finally {
      setIsLoading(false);
    }
  };

  const setMobilePassword = async () => {
    if (newPassword !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    try {
      setIsLoading(true);
      setError(null);
      const res = await fetch(`${API_BASE_URL}/api/set-password-mobile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ password: newPassword }),
      });
      const data: AuthResponse = await res.json();
      if (data.success && data.user) {
        setSuccess('Account created');
        onLogin(data.user);
      } else {
        setError(data.error || 'Failed to set password');
      }
    } catch (e) {
      setError('Unable to set password');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full max-w-full overflow-x-hidden flex flex-col items-center justify-center relative z-10 py-12 px-4">
      {/* Top Header Bar with Back to Home Button */}
      <div className="w-full max-w-4xl flex justify-between items-center mb-8 z-20">
        <button
          onClick={onBackToHome}
          className="inline-flex items-center gap-2.5 px-5 py-2.5 rounded-2xl glass-panel border border-slate-700/80 hover:border-cyan-400/60 text-slate-200 hover:text-cyan-300 text-sm font-semibold transition-all duration-300 shadow-lg group hover:scale-105"
        >
          <ArrowLeft className="w-4 h-4 text-cyan-400 group-hover:-translateX-1 transition-transform" />
          <span>Back to Home</span>
        </button>

        <button
          onClick={onBackToHome}
          className="p-2.5 rounded-2xl glass-panel border border-slate-700/80 hover:border-cyan-400/60 text-slate-300 hover:text-cyan-300 transition-all duration-300 sm:hidden"
          title="Return to Home"
        >
          <Home className="w-4 h-4 text-cyan-400" />
        </button>
      </div>

      {/* Top Brand Logo */}
      <div className="flex flex-col items-center mb-8 space-y-3 text-center">
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-cyan-500 to-indigo-600 flex items-center justify-center shadow-[0_0_30px_rgba(6,182,212,0.4)] border border-cyan-400/40">
          <span className="text-white text-xl font-black tracking-wider">T2D</span>
        </div>
        <div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">
            Insulin <span className="gradient-text-cyan">Predictor Hub</span>
          </h1>
          <p className="text-slate-400 text-xs mt-1 font-light">
            {isLogin ? 'Sign in to access computational drug synthesis' : 'Create an account to start analyzing sequences'}
          </p>
        </div>

        {/* Mode Switcher Tabs */}
        <div className="flex bg-slate-900/80 p-1 rounded-full border border-slate-700/60 backdrop-blur-md">
          <button
            onClick={() => { if (!isLogin) toggleMode(); }}
            className={`px-6 py-2 rounded-full text-xs font-bold transition-all duration-300 ${
              isLogin
                ? 'bg-cyan-500 text-slate-950 shadow-[0_0_15px_rgba(6,182,212,0.4)]'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Sign In
          </button>
          <button
            onClick={() => { if (isLogin) toggleMode(); }}
            className={`px-6 py-2 rounded-full text-xs font-bold transition-all duration-300 ${
              !isLogin
                ? 'bg-purple-500 text-white shadow-[0_0_15px_rgba(168,85,247,0.4)]'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Create Account
          </button>
        </div>
      </div>

      {/* Auth Method Container Cards */}
      <div className="w-full max-w-4xl mb-8">
        <div className="grid md:grid-cols-3 gap-6">
          {/* Method 1: Guest / Username Login */}
          <div
            onClick={() => handleSelectAuth('default')}
            className={`glass-panel rounded-3xl p-6 border transition-all duration-300 cursor-pointer flex flex-col justify-between ${
              selectedAuthMethod === 'default'
                ? 'border-cyan-400 shadow-[0_0_25px_rgba(6,182,212,0.3)] bg-slate-900/90'
                : 'border-slate-800 hover:border-cyan-500/40 opacity-75 hover:opacity-100'
            }`}
          >
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className={`p-3 rounded-2xl ${selectedAuthMethod === 'default' ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40' : 'bg-slate-800 text-slate-400'}`}>
                  <User className="w-6 h-6" />
                </div>
                {selectedAuthMethod === 'default' && (
                  <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping"></span>
                )}
              </div>

              <div>
                <h3 className="text-lg font-bold text-white">Credentials Access</h3>
                <p className="text-xs text-slate-400 font-light mt-1">Sign in with username & password</p>
              </div>

              {selectedAuthMethod === 'default' && (
                <div className="space-y-3 pt-2 animate-fadeIn">
                  <div className="relative">
                    <input
                      type="text"
                      value={username}
                      onChange={(e) => setUsername(e.target.value)}
                      placeholder="Username"
                      disabled={isLoading}
                      onClick={(e) => e.stopPropagation()}
                      className="w-full pl-10 pr-4 py-3 bg-slate-950/80 border border-slate-700 rounded-xl focus:border-cyan-400 text-cyan-300 placeholder-slate-500 text-sm font-mono outline-none transition-all"
                    />
                    <User className="w-4 h-4 text-slate-500 absolute left-3 top-3.5" />
                  </div>

                  <div className="relative">
                    <input
                      type={showPassword ? 'text' : 'password'}
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      placeholder="Password"
                      disabled={isLoading}
                      onClick={(e) => e.stopPropagation()}
                      className="w-full pl-10 pr-10 py-3 bg-slate-950/80 border border-slate-700 rounded-xl focus:border-cyan-400 text-cyan-300 placeholder-slate-500 text-sm font-mono outline-none transition-all"
                    />
                    <Lock className="w-4 h-4 text-slate-500 absolute left-3 top-3.5" />
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setShowPassword((prev) => !prev);
                      }}
                      className="absolute right-3 top-3.5 text-slate-500 hover:text-slate-300"
                    >
                      {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>

                  {!isLogin && password && (
                    <div className="space-y-1 pt-1">
                      <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className={`h-full transition-all duration-300 ${passwordStrength(password).color}`}
                          style={{ width: `${passwordStrength(password).percent}%` }}
                        ></div>
                      </div>
                      <div className="text-[10px] text-slate-400 flex justify-between">
                        <span>Password Strength:</span>
                        <span className="font-bold text-slate-200">{passwordStrength(password).label}</span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Method 2: Google OAuth */}
          <div
            onClick={() => handleSelectAuth('google')}
            className={`glass-panel rounded-3xl p-6 border transition-all duration-300 cursor-pointer flex flex-col justify-between ${
              selectedAuthMethod === 'google'
                ? 'border-purple-400 shadow-[0_0_25px_rgba(168,85,247,0.3)] bg-slate-900/90'
                : 'border-slate-800 hover:border-purple-500/40 opacity-75 hover:opacity-100'
            }`}
          >
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className={`p-3 rounded-2xl ${selectedAuthMethod === 'google' ? 'bg-purple-500/20 text-purple-400 border border-purple-500/40' : 'bg-slate-800 text-slate-400'}`}>
                  <svg className="w-6 h-6" viewBox="0 0 24 24">
                    <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                    <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                    <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                    <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                  </svg>
                </div>
                {selectedAuthMethod === 'google' && (
                  <span className="w-2.5 h-2.5 rounded-full bg-purple-400 animate-ping"></span>
                )}
              </div>

              <div>
                <h3 className="text-lg font-bold text-white">Google OAuth</h3>
                <p className="text-xs text-slate-400 font-light mt-1">Single sign-on via Google Account</p>
              </div>
            </div>
          </div>

          {/* Method 3: Mobile OTP */}
          <div
            onClick={() => handleSelectAuth('mobile')}
            className={`glass-panel rounded-3xl p-6 border transition-all duration-300 cursor-pointer flex flex-col justify-between ${
              selectedAuthMethod === 'mobile'
                ? 'border-emerald-400 shadow-[0_0_25px_rgba(52,211,153,0.3)] bg-slate-900/90'
                : 'border-slate-800 hover:border-emerald-500/40 opacity-75 hover:opacity-100'
            }`}
          >
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className={`p-3 rounded-2xl ${selectedAuthMethod === 'mobile' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40' : 'bg-slate-800 text-slate-400'}`}>
                  <Smartphone className="w-6 h-6" />
                </div>
                {selectedAuthMethod === 'mobile' && (
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping"></span>
                )}
              </div>

              <div>
                <h3 className="text-lg font-bold text-white">Mobile OTP</h3>
                <p className="text-xs text-slate-400 font-light mt-1">Authenticate via SMS passcode</p>
              </div>

              {selectedAuthMethod === 'mobile' && (
                <div className="space-y-3 pt-2 animate-fadeIn" onClick={(e) => e.stopPropagation()}>
                  {mobileStep === 'collect' && (
                    <div className="space-y-2">
                      <input
                        type="text"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        placeholder="Choose a username"
                        className="w-full px-4 py-2.5 bg-slate-950/80 border border-slate-700 rounded-xl text-xs text-emerald-300 placeholder-slate-500 font-mono outline-none"
                      />
                      <div className="flex gap-2">
                        <input
                          type="tel"
                          value={mobile}
                          onChange={(e) => setMobile(e.target.value)}
                          placeholder="Mobile number"
                          className="flex-1 px-4 py-2.5 bg-slate-950/80 border border-slate-700 rounded-xl text-xs text-emerald-300 placeholder-slate-500 font-mono outline-none"
                        />
                        <button
                          type="button"
                          onClick={requestMobileOtp}
                          disabled={!mobile || !username || isLoading}
                          className="px-3 py-2.5 bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold rounded-xl text-xs disabled:opacity-50"
                        >
                          Send
                        </button>
                      </div>
                    </div>
                  )}

                  {mobileStep === 'otp' && (
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={mobileOtp}
                        onChange={(e) => setMobileOtp(e.target.value)}
                        placeholder="Enter OTP Code"
                        className="flex-1 px-4 py-2.5 bg-slate-950/80 border border-slate-700 rounded-xl text-xs text-emerald-300 placeholder-slate-500 font-mono outline-none"
                      />
                      <button
                        type="button"
                        onClick={verifyMobileOtp}
                        disabled={!mobileOtp || isLoading}
                        className="px-4 py-2.5 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold rounded-xl text-xs disabled:opacity-50"
                      >
                        Verify
                      </button>
                    </div>
                  )}

                  {mobileStep === 'setPassword' && (
                    <div className="space-y-2">
                      <input
                        type="password"
                        value={newPassword}
                        onChange={(e) => setNewPassword(e.target.value)}
                        placeholder="Create Password"
                        className="w-full px-4 py-2.5 bg-slate-950/80 border border-slate-700 rounded-xl text-xs text-emerald-300 placeholder-slate-500 font-mono outline-none"
                      />
                      <input
                        type="password"
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        placeholder="Confirm Password"
                        className="w-full px-4 py-2.5 bg-slate-950/80 border border-slate-700 rounded-xl text-xs text-emerald-300 placeholder-slate-500 font-mono outline-none"
                      />
                      <button
                        type="button"
                        onClick={setMobilePassword}
                        disabled={!newPassword || newPassword !== confirmPassword || isLoading}
                        className="w-full py-2.5 bg-emerald-500 text-slate-950 font-bold rounded-xl text-xs"
                      >
                        Set Password & Access
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Notifications */}
      {(error || success) && (
        <div className="w-full max-w-md mb-6">
          {error && (
            <div className="flex items-center gap-3 p-4 bg-red-950/60 border border-red-500/40 rounded-2xl text-red-300 text-xs animate-fadeIn">
              <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
              <p>{error}</p>
            </div>
          )}
          {success && (
            <div className="flex items-center gap-3 p-4 bg-emerald-950/60 border border-emerald-500/40 rounded-2xl text-emerald-300 text-xs animate-fadeIn">
              <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
              <p>{success}</p>
            </div>
          )}
        </div>
      )}

      {/* Main Submit Action Box */}
      <div className="w-full max-w-md">
        <form onSubmit={handleSubmit}>
          {selectedAuthMethod === 'default' && (
            <button
              type="submit"
              disabled={isLoading || !username.trim() || !password.trim()}
              className="btn-neon-cyan w-full py-4 px-6 rounded-2xl text-base font-bold flex items-center justify-center gap-2 shadow-[0_0_20px_rgba(6,182,212,0.4)] transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  <span>{isLogin ? 'Authenticating...' : 'Creating Account...'}</span>
                </>
              ) : (
                <>
                  {isLogin ? <LogIn className="w-5 h-5" /> : <UserPlus className="w-5 h-5" />}
                  <span>{isLogin ? 'Sign In to Suite' : 'Register Account'}</span>
                </>
              )}
            </button>
          )}

          {selectedAuthMethod === 'google' && (
            <button
              type="button"
              onClick={handleGoogleAuth}
              disabled={isLoading}
              className="w-full py-4 px-6 rounded-2xl font-bold bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white flex items-center justify-center gap-3 shadow-[0_0_20px_rgba(168,85,247,0.4)] transition-all"
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24">
                <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
              </svg>
              <span>Continue with Google OAuth</span>
            </button>
          )}
        </form>

        <div className="mt-6 text-center text-xs text-slate-400">
          {isLogin ? "Don't have an account? " : "Already registered? "}
          <button
            type="button"
            onClick={toggleMode}
            className="text-cyan-400 font-bold hover:underline ml-1"
          >
            {isLogin ? 'Create Account' : 'Sign In'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
