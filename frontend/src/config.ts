// Centralized API Base URL configuration for local dev and production (Vercel + Render)
export const API_BASE_URL = (import.meta.env.VITE_API_URL || 'https://insulin-backend-e05j.onrender.com').replace(/\/$/, '');
