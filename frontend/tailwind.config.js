import typography from '@tailwindcss/typography';

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        page: '#F6F1E9',
        surface: '#FFFDF8',
        primary: '#241C15',
        secondary: '#8A7D6C',
        faint: '#B7AB98',
        border: '#E4D9C6',
        accent: '#C4531E',
        link: '#C4531E',
        success: '#4E7A4A',
        warning: '#B8801F',
        danger: '#B23A2C',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '"Microsoft YaHei"', 'sans-serif'],
        display: ['Fraunces', 'Georgia', 'serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      borderRadius: {
        control: '10px',
        panel: '6px',
      },
    },
  },
  plugins: [typography],
};
