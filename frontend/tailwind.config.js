/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // DeepScan Design System - Modern Dark / Deep Slate
        'ds-primary': '#0F172A',      // Main background
        'ds-secondary': '#1E293B',    // Sidebar/Card background
        'ds-elevated': '#334155',     // Hover/Input fields
        'ds-accent': '#3B82F6',       // Primary accent blue
        'ds-accent-hover': '#2563EB', // Accent hover
        'ds-success': '#10B981',      // Real/Authentic
        'ds-danger': '#EF4444',       // Fake/Manipulated
        'ds-warning': '#F59E0B',      // Warning
        'ds-text': '#FFFFFF',         // Primary text
        'ds-text-secondary': '#E2E8F0', // Secondary text
        'ds-text-muted': '#94A3B8',   // Muted text
        'ds-border': '#475569',       // Border color
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
