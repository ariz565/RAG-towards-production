// Centralized UI Configuration

export const SITE_CONFIG = {
  name: "Vision",
  description: "A multi-tenant document intelligence platform.",
  version: "v1.4.2",
};

export const API_CONFIG = {
  baseUrl: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001",
  timeoutMs: 30000,
};

export const ROUTES = {
  overview: "/",
  documents: "/documents",
  chat: "/chat",
  summarize: "/summarize",
  inspector: "/inspector",
  config: "/config",
  account: "/account",
};

export const THEME = {
  colors: {
    paper: "var(--color-paper)",
    paperTint: "var(--color-paper-tint)",
    ink: "var(--color-ink)",
    inkSoft: "var(--color-ink-soft)",
    rule: "var(--color-rule)",
    card: "var(--color-card)",
    cobalt500: "var(--color-cobalt-500)",
  },
  typography: {
    sans: "var(--font-sans)",
    display: "var(--font-display)",
    mono: "var(--font-mono)",
  }
};
