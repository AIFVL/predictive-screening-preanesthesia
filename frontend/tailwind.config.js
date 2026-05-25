/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        fvl: {
          // Verde oscuro institucional (headers, navegación, texto destacado)
          900: "#0f2a1f",
          800: "#143524",
          700: "#1a3c2e",
          600: "#235340",
          // Verde lima brillante (botones de acción, acentos)
          lime: "#7dc242",
          "lime-hover": "#6db035",
          "lime-soft": "#a8d97a",
          // Verde claro (fondos de tarjetas, estados seleccionados)
          mint: "#e8f5e0",
          "mint-strong": "#d4ebc4",
          "mint-border": "#b9d9a3",
          // Neutros tibios para fondo y separadores
          surface: "#f7f9f4",
          line: "#dee5d6",
        },
        risk: {
          low: "#10b981",
          moderate: "#f59e0b",
          elevated: "#f97316",
          high: "#dc2626",
        },
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
      },
      borderRadius: {
        xl: "12px",
        "2xl": "16px",
      },
    },
  },
  plugins: [],
};
