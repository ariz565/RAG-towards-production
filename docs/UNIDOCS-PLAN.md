# UniDocs — The Plan

> *"The people who are crazy enough to think they can change the world are the ones who do."*

---

## The Problem We're Really Solving

Streamlit gave us a prototype. It served its purpose. But here's the truth: **Streamlit is where prototypes go to stay prototypes.** We're not building a prototype anymore — we're building a product.

The real problem isn't "replace Streamlit with React." The real problem is:

**Every document should be conversational. And adding that capability to any website should be as simple as adding Google Analytics — one `<script>` tag.**

That's UniDocs. Not an app. A **platform**.

---

## The Three Deliverables

### 1. `unidocs-widget.js` — The Plugin (The Dent in the Universe)

```html
<!-- This is it. This is the entire integration. -->
<script
  src="https://cdn.unidocs.dev/widget.js"
  data-api="https://api.unidocs.dev"
  data-theme="dark"
  data-accent="#8b5cf6"
  data-greeting="Ask me anything about this document!"
></script>
```

A single JavaScript file. Drop it on any website — WordPress, Shopify, a university homepage, a corporate intranet, a docs site — and instantly get an AI-powered document chatbot. Glassmorphism floating panel. Zero config necessary, infinite config possible.

**Shadow DOM** for complete style isolation. Your website's CSS can't break it. Its CSS can't break your website. They coexist peacefully.

### 2. The Standalone App — `unidocs.dev`

A full-page web experience for when UniDocs IS the product:
- **Landing page**: Glassmorphism hero that makes you want to try it
- **Chat page**: Full-screen document conversation with citations, pipeline trace, tree explorer
- **Dashboard**: Upload PDFs, build indexes, configure providers and strategies

### 3. The Shared Core — Written Once, Used Everywhere

The chat interface, API client, streaming logic, message rendering — **one codebase** powering both the widget and the standalone app. Change it once, it improves everywhere.

---

## Why This Architecture Is Inevitable

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│    ANY WEBSITE                    STANDALONE APP             │
│    ──────────                     ──────────────             │
│    <script src="widget.js">       unidocs.dev/chat           │
│           │                             │                    │
│           ▼                             ▼                    │
│    ┌─────────────┐              ┌──────────────┐             │
│    │   Widget     │              │   App Shell   │            │
│    │  (Shadow DOM)│              │  (Full Page)  │            │
│    └──────┬──────┘              └──────┬───────┘             │
│           │                            │                     │
│           └────────────┬───────────────┘                     │
│                        ▼                                     │
│              ┌──────────────────┐                            │
│              │   SHARED CORE    │                            │
│              │                  │                            │
│              │  ChatPanel       │                            │
│              │  MessageBubble   │                            │
│              │  TypingIndicator │                            │
│              │  CitationCard    │                            │
│              │  PipelineTrace   │                            │
│              │  TreeExplorer    │                            │
│              │  API Client      │                            │
│              │  SSE Streaming   │                            │
│              │  Chat Store      │                            │
│              │  Glass* UI       │                            │
│              └────────┬─────────┘                            │
│                       │                                      │
│                       ▼                                      │
│              ┌──────────────────┐                            │
│              │   FastAPI Backend │                            │
│              │                  │                            │
│              │  POST /api/ask   │  ← Standard Q&A            │
│              │  POST /ask/stream│  ← SSE real-time           │
│              │  GET  /api/tree  │  ← Document structure      │
│              │  GET  /api/page  │  ← Raw page text           │
│              │  POST /api/index │  ← Build indexes           │
│              │  *  /api/config  │  ← Runtime switching       │
│              │  GET  /api/health│  ← Status check            │
│              └──────────────────┘                            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

Why is this the *only* architecture that makes sense?

1. **The widget IS the chat panel.** If we build the chat panel well, the widget is just a floating container around it. One component. Two surfaces.

2. **The API already exists.** We built a complete REST + SSE streaming backend. The frontend is just a beautiful skin over it.

3. **Shadow DOM is non-negotiable for plugins.** Any CSS-in-JS or scoped-styles approach is fragile at scale. Shadow DOM is the browser's own isolation primitive. We use it.

4. **React + Vite can produce both a SPA and an IIFE library** from the same source tree. Two `vite.config` files. Same components. Different entry points.

---

## The Design Language: "Aurora Glass"

Inspired by context7.com's dark aesthetic, but evolved into something that feels uniquely ours.

### Philosophy

Glass isn't decoration. It's **information hierarchy**. The translucency tells you what's foreground and what's background. The blur tells you there's depth. The subtle borders tell you where one surface ends and another begins. Every pixel has a purpose.

### The Canvas

```
Background: #050505 (near-black)

Aurora mesh: Three gradient orbs that shift slowly (15s cycle)
  ┌─ Orb 1: violet-600/20  (top-left, 600px, blur 150px)
  ├─ Orb 2: cyan-500/15    (bottom-right, 500px, blur 120px)
  └─ Orb 3: emerald-500/10 (center-bottom, 400px, blur 100px)

Motion: CSS @keyframes, GPU-accelerated transforms only
        Subtle enough to feel alive, not enough to distract
```

### Glass Surfaces (4 Elevation Levels)

```css
/* Level 0 — Sunken (input fields, code blocks) */
.glass-sunken {
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.04);
  backdrop-filter: blur(8px);
}

/* Level 1 — Base (cards, panels, messages) */
.glass-base {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.07);
  backdrop-filter: blur(24px);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

/* Level 2 — Elevated (floating panels, dropdowns, widget) */
.glass-elevated {
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.10);
  backdrop-filter: blur(40px);
  box-shadow: 0 16px 48px rgba(0, 0, 0, 0.4);
}

/* Level 3 — Prominent (modal overlays, active selections) */
.glass-prominent {
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.14);
  backdrop-filter: blur(48px);
  box-shadow: 0 24px 64px rgba(0, 0, 0, 0.5);
}
```

### Color System

```
┌─────────────────────────────────────────────────┐
│  ACCENT GRADIENT                                │
│  from-violet-500  via-purple-500  to-cyan-400   │
│  Used for: primary button, send button,         │
│            active states, loading indicators,    │
│            widget FAB, link hover                │
│                                                  │
│  TEXT HIERARCHY                                  │
│  white/90  — Headlines, user messages            │
│  white/70  — Body text, AI responses             │
│  white/50  — Secondary info, timestamps          │
│  white/30  — Muted labels, placeholders          │
│  white/15  — Disabled, dividers                  │
│                                                  │
│  SEMANTIC                                        │
│  emerald-400/80  — Success, confidence > 0.8     │
│  amber-400/80    — Warning, confidence 0.5-0.8   │
│  red-400/80      — Error, confidence < 0.5       │
│  violet-400/80   — Info, citations, thinking     │
│  cyan-400/80     — Highlight, active node        │
│                                                  │
│  GRADIENTS                                       │
│  User bubble: from-violet-600 to-purple-600      │
│  AI thinking: from-violet-500/5 to-cyan-500/5    │
│  Section headers: from-white/10 to-transparent   │
└─────────────────────────────────────────────────┘
```

### Typography

```
Font: Inter (variable, 400/500/600/700)
      Fallback: system-ui, -apple-system, sans-serif

Scale:
  xs:   11px / 1.4   — Timestamps, badges
  sm:   13px / 1.5   — Secondary text, captions
  base: 14px / 1.6   — Body text, messages
  lg:   16px / 1.5   — Section headers
  xl:   20px / 1.3   — Page titles
  2xl:  28px / 1.2   — Hero subtitle
  3xl:  40px / 1.1   — Hero headline
  4xl:  56px / 1.05  — Landing hero

Code: JetBrains Mono, 13px
```

### Spacing & Radius

```
Card radius:    16px (rounded-2xl)
Button radius:  12px (rounded-xl)
Input radius:   12px (rounded-xl)
Badge radius:   8px  (rounded-lg)
Avatar radius:  full (rounded-full)
Widget radius:  20px (rounded-[20px])

Message gap:    8px
Section gap:    24px
Page padding:   24px (mobile: 16px)
```

---

## Component Architecture

### The Shared Core (`src/unidocs/`)

```
src/unidocs/
├── components/
│   ├── glass/                        # Glassmorphism primitives
│   │   ├── GlassCard.tsx             # Base glass surface
│   │   ├── GlassInput.tsx            # Text input with glass styling
│   │   ├── GlassButton.tsx           # Button variants (solid, ghost, gradient)
│   │   ├── GlassSelect.tsx           # Dropdown with glass panel
│   │   ├── GlassTooltip.tsx          # Tooltip with glass effect
│   │   └── AuroraBackground.tsx      # Animated gradient mesh background
│   │
│   ├── chat/                          # Chat components
│   │   ├── ChatPanel.tsx              # ★ THE CORE — message list + input + state
│   │   ├── ChatInput.tsx              # Multi-line input with send button + attachments
│   │   ├── MessageBubble.tsx          # Single message (user or assistant)
│   │   ├── AssistantMessage.tsx       # AI response with markdown + citations inline
│   │   ├── UserMessage.tsx            # User message (gradient pill)
│   │   ├── TypingIndicator.tsx        # Three-dot pulse + optional "thinking" text
│   │   ├── WelcomeScreen.tsx          # Empty state — logo, greeting, suggested Qs
│   │   ├── SuggestedQuestions.tsx     # Clickable question chips
│   │   └── ChatHeader.tsx            # Title + config toggles + minimize button
│   │
│   ├── citations/                     # Source attribution
│   │   ├── CitationCard.tsx           # Expandable citation with page preview
│   │   ├── CitationBadge.tsx          # Compact inline [1] badge
│   │   └── CitationPanel.tsx          # Side panel with all sources
│   │
│   ├── pipeline/                      # Observability
│   │   ├── PipelineTrace.tsx          # Collapsible step-by-step trace
│   │   ├── PipelineStep.tsx           # Single step with status + timing
│   │   └── PipelineMetrics.tsx        # Token count, duration, confidence
│   │
│   └── tree/                          # Document structure
│       ├── TreeExplorer.tsx           # Recursive tree with expand/collapse
│       └── TreeNode.tsx               # Single node with page range + summary
│
├── hooks/
│   ├── useChat.ts                     # Chat message state + send/receive logic
│   ├── useStreaming.ts                # SSE connection + event parsing
│   ├── useConfig.ts                   # Backend config (providers, strategies)
│   └── useHealth.ts                   # Backend health check polling
│
├── api/
│   └── client.ts                      # Typed fetch wrapper for all endpoints
│                                        ask(), askStream(), getTree(), getPage(),
│                                        getConfig(), updateConfig(), indexDocument(),
│                                        health()
│
├── store/
│   └── chatStore.ts                   # Zustand: messages, isStreaming, config,
│                                        conversation history, selected citations
│
├── types/
│   └── index.ts                       # All shared types (mirroring backend schemas)
│                                        Message, Citation, PipelineStep, TreeNode,
│                                        AskRequest, AskResponse, StreamEvent, etc.
│
└── styles/
    └── glass.css                      # Glassmorphism utility classes
                                         (injected into Shadow DOM for widget)
```

### The Standalone App (`src/app/`)

```
src/app/
├── pages/
│   ├── Landing.tsx                    # Hero + features + social proof + CTA
│   ├── Chat.tsx                       # Full-page chat (ChatPanel + sidebar)
│   ├── Dashboard.tsx                  # Upload, index, manage documents
│   └── Embed.tsx                      # Widget configurator + embed code generator
│
├── layout/
│   ├── AppShell.tsx                   # Top nav + main content area
│   ├── ChatLayout.tsx                 # Chat-specific layout (sidebar + main)
│   └── Sidebar.tsx                    # Tree explorer + config + page viewer
│
└── routes.tsx                         # React Router configuration
```

### The Widget (`src/widget/`)

```
src/widget/
├── Widget.tsx                         # Root: FAB + Panel + state orchestration
├── WidgetButton.tsx                   # Floating action button (bottom-right)
├── WidgetPanel.tsx                    # Expandable chat panel (glassmorphism)
├── WidgetConfig.ts                    # Parse data-* attributes + JS API
├── mount.tsx                          # Create Shadow DOM, inject styles, render React
└── entry.ts                           # IIFE entry: window.UniDocs = { init() }
```

---

## The Chat Experience — In Detail

### Message Flow

```
User types question
       │
       ▼
┌─────────────────────┐
│   ChatInput          │  Shift+Enter = newline, Enter = send
│   [textarea]  [➤]   │  Character counter, paste support
└──────────┬──────────┘
           │
           ▼
     POST /api/ask/stream  (SSE)
           │
           ├── event: node_started    → PipelineStep appears (spinner)
           ├── event: node_thinking   → Thinking text streams in
           ├── event: node_completed  → Step gets checkmark + timing
           ├── event: answer_chunk    → Answer text streams word-by-word
           └── event: pipeline_complete → Final answer + citations + metrics
                                          │
                                          ▼
                                ┌──────────────────┐
                                │ AssistantMessage  │
                                │                  │
                                │  Markdown answer │
                                │  ─────────────── │
                                │  📄 Citation [1] │
                                │  📄 Citation [2] │
                                │  ─────────────── │
                                │  ▸ Pipeline trace │
                                │  ─────────────── │
                                │  ⏱ 2.3s  💬 847  │
                                └──────────────────┘
```

### Message Anatomy

**User Message:**
```
┌─────────────────────────────────────────────┐
│          ┌──────────────────────────┐       │
│          │ What GPA do I need for   │ ←─── gradient pill
│          │ Computer Science?        │       violet → purple
│          └──────────────────────────┘       │
│                              2:34 PM        │
└─────────────────────────────────────────────┘
```

**Assistant Message:**
```
┌─────────────────────────────────────────────┐
│  🤖                                          │
│  ┌────────────────────────────────────────┐  │
│  │ For the **Computer Science** program,  │  │
│  │ the minimum GPA requirement is **3.2** │  │ ←─── glass card
│  │ on a 4.0 scale. Transfer students...   │  │       left border gradient
│  │                                        │  │
│  │ ┌──────┐ ┌──────┐                     │  │
│  │ │ p.12 │ │ p.45 │  ← citation badges  │  │
│  │ └──────┘ └──────┘                     │  │
│  │                                        │  │
│  │ ▸ View pipeline (6 steps, 2.3s)       │  │ ←─── expandable
│  │                                        │  │
│  │  ◉ 92% confident  ·  847 tokens       │  │ ←─── footer metrics
│  └────────────────────────────────────────┘  │
│                              2:34 PM         │
└─────────────────────────────────────────────┘
```

### Welcome Screen (Empty State)

```
┌─────────────────────────────────────────────────┐
│                                                 │
│              ┌─────┐                            │
│              │ 📚  │   ← animated logo          │
│              └─────┘                            │
│                                                 │
│          Welcome to UniDocs                     │
│                                                 │
│    Ask anything about your documents.           │
│    I'll find the answer with citations.         │
│                                                 │
│   ┌─────────────────────────────────────┐       │
│   │ What are the admission requirements │       │ ←─── suggestion chips
│   │ for Computer Science?               │       │      (glass cards)
│   └─────────────────────────────────────┘       │
│   ┌─────────────────────────────────────┐       │
│   │ What scholarships are available?    │       │
│   └─────────────────────────────────────┘       │
│   ┌─────────────────────────────────────┐       │
│   │ What documents do I need to submit? │       │
│   └─────────────────────────────────────┘       │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## The Widget — Technical Deep Dive

### How It Works

```javascript
// 1. Script loads on any webpage
// 2. Reads data-* attributes for configuration
// 3. Creates a Shadow DOM container (CSS isolation)
// 4. Injects scoped Tailwind CSS into shadow root
// 5. Renders React app inside shadow root
// 6. Exposes window.UniDocs API for programmatic control

// Host page HTML:
<script
  src="/unidocs-widget.js"
  data-api="http://localhost:8001"
  data-theme="dark"
  data-position="bottom-right"
  data-accent="#8b5cf6"
  data-greeting="Ask me about admissions!"
  data-strategy="hybrid"
  data-provider="azure_openai"
></script>

// Or programmatic:
window.UniDocs.init({
  apiUrl: 'http://localhost:8001',
  theme: 'dark',
  position: 'bottom-right',
  accent: '#8b5cf6',
  greeting: 'Ask me about admissions!',
  strategy: 'hybrid',
  provider: 'azure_openai',
});

// Control API:
window.UniDocs.open();       // Open chat panel
window.UniDocs.close();      // Close chat panel
window.UniDocs.toggle();     // Toggle panel
window.UniDocs.send('Hi');   // Send a message programmatically
window.UniDocs.destroy();    // Remove widget from page
```

### Shadow DOM Mount Strategy

```typescript
// mount.tsx — The isolation layer

function mountWidget(config: WidgetConfig) {
  // 1. Create host element
  const host = document.createElement('div');
  host.id = 'unidocs-widget-host';
  host.style.cssText = 'position:fixed;z-index:2147483647;'; // Max z-index
  document.body.appendChild(host);

  // 2. Attach shadow root
  const shadow = host.attachShadow({ mode: 'open' });

  // 3. Inject scoped styles (Tailwind subset + glass.css)
  const style = document.createElement('style');
  style.textContent = SCOPED_CSS; // Built at compile time
  shadow.appendChild(style);

  // 4. Create React root inside shadow DOM
  const container = document.createElement('div');
  container.id = 'unidocs-root';
  shadow.appendChild(container);

  // 5. Render React
  const root = createRoot(container);
  root.render(<Widget config={config} />);

  return { root, host, shadow };
}
```

### Widget Visual States

```
STATE 1: Closed (just the button)
┌──────┐
│  💬  │  ← 56px circle, gradient bg, bottom-right
└──────┘    Subtle pulse animation when idle
            Badge with unread count if messages waiting

STATE 2: Open (button + panel)
                        ┌──────────────────────────┐
                        │ UniDocs           ─  ✕   │ ← header
                        ├──────────────────────────┤
                        │                          │
                        │   Welcome to UniDocs     │ ← chat area
                        │   Ask anything...        │    (scrollable)
                        │                          │
                        │   ┌──────────────────┐   │
                        │   │ Suggested Q1     │   │
                        │   └──────────────────┘   │
                        │   ┌──────────────────┐   │
                        │   │ Suggested Q2     │   │
                        │   └──────────────────┘   │
                        │                          │
                        ├──────────────────────────┤
                        │ Ask a question...    [➤] │ ← input
                        └──────────────────────────┘
┌──────┐
│  ✕   │  ← Button becomes close icon
└──────┘

STATE 3: Streaming (mid-response)
                        ┌──────────────────────────┐
                        │ UniDocs           ─  ✕   │
                        ├──────────────────────────┤
                        │                          │
                        │  ┌────────────────────┐  │
                        │  │ User question      │  │
                        │  └────────────────────┘  │
                        │                          │
                        │  🤖 ┌─────────────────┐  │
                        │     │ The GPA require- │  │
                        │     │ ment is ▌        │  │ ← streaming cursor
                        │     │                  │  │
                        │     │ ● ● ●            │  │ ← thinking dots
                        │     └─────────────────┘  │
                        │                          │
                        ├──────────────────────────┤
                        │ ████████████ Thinking... │ ← disabled during stream
                        └──────────────────────────┘
```

### Widget Dimensions & Responsiveness

```
Desktop:
  Button: 56px × 56px
  Panel:  400px × min(600px, calc(100vh - 100px))
  Margin: 20px from edges

Tablet:
  Button: 52px × 52px
  Panel:  380px × min(580px, calc(100vh - 80px))
  Margin: 16px from edges

Mobile (< 640px):
  Button: 48px × 48px
  Panel:  100vw × 100vh (fullscreen takeover)
  Margin: 0 (edge-to-edge)
  Header gets back arrow instead of minimize
```

---

## Standalone App — Page Designs

### Landing Page (`/`)

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─ Nav ──────────────────────────────────────────────────┐ │
│  │  🔮 UniDocs          Features   Pricing   Try It  [→] │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  ──── Aurora gradient mesh background ────                  │
│                                                             │
│           Your documents,                                   │
│           finally searchable.                               │ ← hero
│                                                             │
│     Drop a PDF. Ask a question. Get cited answers.          │
│                                                             │
│     [  Try UniDocs  →  ]    [ View Demo ]                   │
│                                                             │
│                                                             │
│  ─── Glass demo card ───────────────────────────────────    │
│  │  Live chat demo (embedded ChatPanel, pre-loaded msgs) │  │
│  │  Or animated mockup showing a question → answer flow  │  │
│  ────────────────────────────────────────────────────────   │
│                                                             │
│                                                             │
│  ─── Features Grid (3 cards) ───────────────────────────    │
│  │ 🧠 Multi-Strategy RAG │ 🔌 One-Line Embed  │ 🎯 Cited │ │
│  │ PageIndex, Hybrid,    │ Works on any site  │ Every    │  │
│  │ BM25, Vector — pick   │ as a chatbot       │ answer   │  │
│  │ what works best.      │ plugin.            │ has page │  │
│  │                       │                    │ numbers. │  │
│  ────────────────────────────────────────────────────────   │
│                                                             │
│  ─── Provider logos strip ──────────────────────────────    │
│  │ OpenAI · Azure · Groq · Gemini · OpenRouter · Ollama │  │
│  ────────────────────────────────────────────────────────   │
│                                                             │
│  ─── How It Works (3 steps) ───────────────────────────     │
│  │ 1. Upload PDF  →  2. Build Index  →  3. Ask Away    │   │
│  ────────────────────────────────────────────────────────   │
│                                                             │
│  ─── Embed CTA ────────────────────────────────────────     │
│  │  Add UniDocs to your website in 30 seconds.          │  │
│  │  <script src="unidocs-widget.js"></script>            │  │
│  │  [ Get Widget Code → ]                               │  │
│  ────────────────────────────────────────────────────────   │
│                                                             │
│  ─── Footer ───────────────────────────────────────────     │
│  │  🔮 UniDocs   ·   GitHub   ·   Docs   ·   API       │  │
│  ────────────────────────────────────────────────────────   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Chat Page (`/chat`)

```
┌─────────────────────────────────────────────────────────────┐
│ 🔮 UniDocs  │  New Chat  │ ⚙ Settings │     model: gpt-4o │
├─────────────┼───────────────────────────────────────────────┤
│             │                                               │
│  SIDEBAR    │           CHAT AREA                           │
│  (280px)    │                                               │
│             │   🤖 Welcome to UniDocs                       │
│ ▾ Document  │                                               │
│   Structure │   Ask anything about:                         │
│   ├ Ch. 1   │   📄 Admission Guide 2025                     │
│   ├ Ch. 2   │                                               │
│   │ ├ 2.1   │   ┌────────────────────────────────────────┐  │
│   │ ├ 2.2   │   │ What are the GPA requirements?       │  │
│   │ └ 2.3   │   └────────────────────────────────────────┘  │
│   ├ Ch. 3   │   ┌────────────────────────────────────────┐  │
│   └ Ch. 4   │   │ What scholarships are available?      │  │
│             │   └────────────────────────────────────────┘  │
│ ──────────  │   ┌────────────────────────────────────────┐  │
│             │   │ What documents do I need?              │  │
│ ▾ Config    │   └────────────────────────────────────────┘  │
│  Strategy:  │                                               │
│  [Hybrid ▾] │                                               │
│  Provider:  │                                               │
│  [Azure  ▾] │                                               │
│             │                                               │
│ ──────────  │                                               │
│ 📊 Stats    │ ┌──────────────────────────────────────────┐  │
│ Pages: 150  │ │ Ask a question...                    [➤] │  │
│ Strategy:   │ └──────────────────────────────────────────┘  │
│ hybrid      │                                               │
│             │  Strategy: Hybrid  │  Model: GPT-4o-mini      │
├─────────────┴───────────────────────────────────────────────┤
│  ◉ Connected  ·  150 pages indexed  ·  Hybrid + Azure      │
└─────────────────────────────────────────────────────────────┘
```

### Dashboard Page (`/dashboard`)

```
┌─────────────────────────────────────────────────────────────┐
│ 🔮 UniDocs  │  Chat  │  Dashboard  │  Embed               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Documents                                                  │
│  ──────────────────────────────────────                     │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  ╔═══════════════════════════════════════════════╗   │   │
│  │  ║                                               ║   │   │
│  │  ║        ┌───────────────┐                      ║   │   │
│  │  ║        │  📄 Drop PDF  │                      ║   │   │
│  │  ║        │  here or      │                      ║   │   │
│  │  ║        │  click to     │                      ║   │   │
│  │  ║        │  browse       │                      ║   │   │
│  │  ║        └───────────────┘                      ║   │   │
│  │  ║                                               ║   │   │
│  │  ╚═══════════════════════════════════════════════╝   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  Indexed Documents                                          │
│  ──────────────────────────────────────                     │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 📄 Admission Guide 2025                              │   │
│  │    150 pages · 42 tree nodes · 312 chunks            │   │
│  │    PageIndex ✓  Hybrid ✓                             │   │
│  │    [ Chat → ]   [ Rebuild ]   [ Delete ]             │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  System Configuration                                       │
│  ──────────────────────────────────────                     │
│                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │ LLM Provider  │  │ Strategy      │  │ Embeddings    │   │
│  │ [Azure OAI ▾] │  │ [Hybrid    ▾] │  │ [HuggingFace▾]│  │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Backend Health                                       │   │
│  │ Status: ● Connected    API: http://localhost:8001    │   │
│  │ Uptime: 2h 34m        Queries: 47                   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Backend Enhancements

The current FastAPI backend is already 90% there. We need minimal additions:

### 1. File Upload Endpoint

```python
# POST /api/upload — Accept PDF via multipart form
@router.post("/api/upload")
async def upload_pdf(file: UploadFile) -> IndexResponse:
    # Save to data/pdf/
    # Optionally trigger indexing
    pass
```

### 2. Widget Configuration Endpoint

```python
# GET /api/widget/config — Returns widget appearance settings
@router.get("/api/widget/config")
async def widget_config() -> dict:
    return {
        "greeting": settings.widget_greeting,
        "suggested_questions": settings.widget_suggestions,
        "theme": "dark",
        "accent": "#8b5cf6",
        "doc_name": document_index.doc_name,
    }
```

### 3. CORS Update for Widget

```python
# Allow any origin for the widget (or use API key auth)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Widget can be on any domain
    allow_methods=["GET", "POST"],
    allow_headers=["*", "X-UniDocs-Key"],  # API key header
)
```

### 4. Optional: API Key Middleware

```python
# Simple API key validation for widget usage
@app.middleware("http")
async def validate_api_key(request, call_next):
    if request.url.path.startswith("/api/") and "X-UniDocs-Key" in request.headers:
        # Validate key
        pass
    return await call_next(request)
```

---

## File Structure — Complete

```
agent-frontend/
├── public/
│   └── robots.txt
│
├── src/
│   ├── unidocs/                           # ★ SHARED CORE ★
│   │   ├── components/
│   │   │   ├── glass/
│   │   │   │   ├── AuroraBackground.tsx   # Animated gradient mesh
│   │   │   │   ├── GlassCard.tsx          # Base glass card
│   │   │   │   ├── GlassInput.tsx         # Glass text input
│   │   │   │   ├── GlassButton.tsx        # Glass button (variants)
│   │   │   │   ├── GlassSelect.tsx        # Glass dropdown
│   │   │   │   └── index.ts              # Barrel export
│   │   │   │
│   │   │   ├── chat/
│   │   │   │   ├── ChatPanel.tsx          # ★ Core chat component
│   │   │   │   ├── ChatInput.tsx          # Message input bar
│   │   │   │   ├── ChatHeader.tsx         # Panel header
│   │   │   │   ├── MessageList.tsx        # Scrollable message list
│   │   │   │   ├── MessageBubble.tsx      # Dispatch to User/Assistant
│   │   │   │   ├── UserMessage.tsx        # Gradient pill
│   │   │   │   ├── AssistantMessage.tsx   # Glass card + markdown
│   │   │   │   ├── TypingIndicator.tsx    # Animated thinking dots
│   │   │   │   ├── WelcomeScreen.tsx      # Empty state
│   │   │   │   ├── SuggestedQuestions.tsx  # Question chips
│   │   │   │   └── index.ts
│   │   │   │
│   │   │   ├── citations/
│   │   │   │   ├── CitationBadge.tsx      # Inline [p.12] badge
│   │   │   │   ├── CitationCard.tsx       # Expandable source card
│   │   │   │   ├── CitationPanel.tsx      # Citations sidebar/section
│   │   │   │   └── index.ts
│   │   │   │
│   │   │   ├── pipeline/
│   │   │   │   ├── PipelineTrace.tsx      # Accordion trace viewer
│   │   │   │   ├── PipelineStep.tsx       # Single step (status icon + timing)
│   │   │   │   ├── PipelineMetrics.tsx    # Summary bar (tokens, time, confidence)
│   │   │   │   └── index.ts
│   │   │   │
│   │   │   └── tree/
│   │   │       ├── TreeExplorer.tsx       # Document tree browser
│   │   │       ├── TreeNode.tsx           # Recursive tree node
│   │   │       └── index.ts
│   │   │
│   │   ├── hooks/
│   │   │   ├── useChat.ts                # Message state + conversation flow
│   │   │   ├── useStreaming.ts           # SSE event source + parsing
│   │   │   ├── useConfig.ts             # Read/update backend config
│   │   │   └── useHealth.ts             # Health check polling
│   │   │
│   │   ├── api/
│   │   │   └── client.ts                # Typed fetch: ask, stream, tree, page, etc.
│   │   │
│   │   ├── store/
│   │   │   └── chatStore.ts             # Zustand: messages[], isStreaming, config
│   │   │
│   │   ├── types/
│   │   │   └── index.ts                 # Message, Citation, PipelineStep, etc.
│   │   │
│   │   └── styles/
│   │       └── glass.css                # Glassmorphism utility classes
│   │
│   ├── app/                              # ★ STANDALONE APP ★
│   │   ├── pages/
│   │   │   ├── Landing.tsx               # Marketing landing page
│   │   │   ├── Chat.tsx                  # Full-page chat experience
│   │   │   ├── Dashboard.tsx             # Document management
│   │   │   └── Embed.tsx                 # Widget code generator
│   │   │
│   │   ├── layout/
│   │   │   ├── AppShell.tsx              # Nav + content wrapper
│   │   │   ├── ChatLayout.tsx            # Sidebar + chat area
│   │   │   └── Sidebar.tsx               # Tree + config + stats
│   │   │
│   │   └── routes.tsx                    # Router config
│   │
│   ├── widget/                           # ★ EMBEDDABLE WIDGET ★
│   │   ├── Widget.tsx                    # Widget orchestrator
│   │   ├── WidgetButton.tsx              # FAB button
│   │   ├── WidgetPanel.tsx               # Chat panel container
│   │   ├── WidgetConfig.ts              # Config parser (data-* attrs + JS)
│   │   ├── mount.tsx                    # Shadow DOM mount
│   │   └── entry.ts                     # IIFE entry (window.UniDocs)
│   │
│   ├── main.tsx                          # Standalone app entry point
│   └── widget-entry.tsx                  # Widget build entry point
│
├── index.html                            # SPA HTML shell
├── package.json
├── vite.config.ts                        # Main app build config
├── vite.widget.config.ts                 # Widget IIFE build config
├── tailwind.config.ts                    # Extended with glass utilities
├── tsconfig.json
└── CLAUDE.md                             # Guiding principles
```

---

## Build Strategy

### Two Vite Configs, One Codebase

**`vite.config.ts`** — Standalone App (SPA)
```typescript
export default defineConfig({
  build: {
    outDir: 'dist/app',
    rollupOptions: {
      input: 'index.html',
    },
  },
});
```

**`vite.widget.config.ts`** — Widget (IIFE Library)
```typescript
export default defineConfig({
  build: {
    outDir: 'dist/widget',
    lib: {
      entry: 'src/widget/entry.ts',
      name: 'UniDocs',
      formats: ['iife'],
      fileName: () => 'unidocs-widget.js',
    },
    rollupOptions: {
      // Bundle React inside (don't externalize)
      external: [],
    },
    cssCodeSplit: false,  // Single CSS string for shadow DOM injection
  },
});
```

### NPM Scripts

```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "build:widget": "vite build --config vite.widget.config.ts",
    "build:all": "npm run build && npm run build:widget",
    "preview": "vite preview"
  }
}
```

---

## Phased Implementation Plan

### Phase 1: Foundation (Days 1-2)
**Goal: Glass design system + typed API client + basic chat working**

- [ ] Set up glassmorphism Tailwind config (glass utilities, aurora keyframes)
- [ ] Build `GlassCard`, `GlassInput`, `GlassButton`, `AuroraBackground`
- [ ] Build typed API client (`client.ts`) wrapping all backend endpoints
- [ ] Build `chatStore.ts` (Zustand) with message management
- [ ] Build `useChat` hook (send message, receive response)
- [ ] Build `useStreaming` hook (SSE connection, event parsing)
- [ ] Build `ChatInput`, `MessageBubble`, `UserMessage`, `AssistantMessage`
- [ ] Build `TypingIndicator` with three-dot pulse animation
- [ ] Build `ChatPanel` composing all chat sub-components
- [ ] Build `WelcomeScreen` with `SuggestedQuestions`
- [ ] Wire up to backend: send question → receive streamed answer

### Phase 2: Rich Features (Days 3-4)
**Goal: Citations, pipeline trace, tree explorer, config switching**

- [ ] Build `CitationBadge` and `CitationCard`
- [ ] Build `PipelineTrace` with `PipelineStep` accordion
- [ ] Build `PipelineMetrics` bar (tokens, duration, confidence)
- [ ] Build `TreeExplorer` and `TreeNode` (recursive)
- [ ] Build `useConfig` hook + config switching UI
- [ ] Build `useHealth` hook + connection status indicator
- [ ] Integrate citations into `AssistantMessage`
- [ ] Add markdown rendering (react-markdown + syntax highlighting)
- [ ] Add message animations (framer-motion entrance/exit)

### Phase 3: Standalone App Pages (Days 5-6)
**Goal: Full multi-page app with landing, chat, dashboard, embed**

- [ ] Build `AppShell` layout with navigation
- [ ] Build `Landing` page (hero, features, CTA)
- [ ] Build `Chat` page with `ChatLayout` (sidebar + main area)
- [ ] Build `Sidebar` (tree explorer + config panel + stats)
- [ ] Build `Dashboard` page (upload, index, manage documents)
- [ ] Build `Embed` page (widget configurator + code generator)
- [ ] Add React Router routes
- [ ] Add backend file upload endpoint (`POST /api/upload`)
- [ ] Add page transitions (framer-motion AnimatePresence)

### Phase 4: Widget (Days 7-8)
**Goal: Embeddable IIFE widget with Shadow DOM isolation**

- [ ] Build `WidgetConfig` (data-* parser + JS API)
- [ ] Build `WidgetButton` (floating action button)
- [ ] Build `WidgetPanel` (glassmorphism chat container)
- [ ] Build `Widget` orchestrator (open/close state, animations)
- [ ] Build `mount.tsx` (Shadow DOM creation + style injection)
- [ ] Build `entry.ts` (window.UniDocs API surface)
- [ ] Create `vite.widget.config.ts` (IIFE build)
- [ ] Extract + scope Tailwind CSS for shadow DOM
- [ ] Test on plain HTML page
- [ ] Test on external website simulation
- [ ] Mobile responsive (fullscreen takeover on small screens)

### Phase 5: Polish & Ship (Days 9-10)
**Goal: Production-ready, delightful, accessible**

- [ ] Keyboard navigation (Cmd+K to focus input, Escape to close)
- [ ] Accessibility audit (ARIA labels, focus management, screen reader)
- [ ] Error states (network error, API error, empty results)
- [ ] Loading skeletons for initial data fetches
- [ ] Dark/light theme toggle (for standalone app)
- [ ] Performance optimization (React.memo, virtualized message list)
- [ ] Widget CSS minification (< 50KB)
- [ ] End-to-end testing
- [ ] Update CLAUDE.md with new architecture
- [ ] Documentation (README, API docs, widget integration guide)

---

## Key Decisions & Rationale

| Decision | Why |
|----------|-----|
| **Shadow DOM for widget** | Only browser-native CSS isolation. CSS-in-JS leaks. Scoped styles are fragile. Shadow DOM is the spec. |
| **Zustand over Redux** | Already in the codebase. Simpler API. No boilerplate. Perfect for chat state. |
| **SSE over WebSocket** | Backend already implements SSE. Simpler to implement. Auto-reconnect with EventSource. Sufficient for unidirectional streaming. |
| **Two Vite configs** | Clean separation. Widget gets tree-shaken IIFE. App gets code-split SPA. Same source, different targets. |
| **Inter font** | Industry-standard for modern UIs. Excellent readability. Variable font = one HTTP request. |
| **Shared core as local package** | No npm publish overhead. Import directly from `@/unidocs/`. Single source of truth. |
| **Framer Motion for animations** | Already in the codebase. Spring physics feel natural. AnimatePresence for exit animations. |
| **react-markdown for AI responses** | Standard solution. Code block highlighting. Table support. Link handling. |
| **Tailwind glassmorphism utilities** | Consistent glass effects via utility classes. No duplicated CSS. Easy to maintain. |

---

## Success Metrics

When this is done, you should be able to:

1. **Open `unidocs.dev/`** — See a landing page that makes you want to try the product
2. **Open `unidocs.dev/chat`** — Ask a question, watch it stream in with citations and pipeline trace
3. **Open any HTML file** — Drop in `<script src="unidocs-widget.js">`, see a floating chat button, click it, ask a question, get a cited answer
4. **Switch providers mid-conversation** — Toggle from Azure to Groq in the sidebar, next message uses Groq
5. **Switch strategies** — Toggle from Hybrid to PageIndex, see different retrieval paths in the pipeline trace
6. **View the document tree** — Navigate the hierarchical structure, click a node to jump to that section
7. **Feel the glass** — Every surface has depth, blur, and translucency. The aurora background breathes life into the dark canvas.

---

*"Design is not just what it looks like and feels like. Design is how it works."*

— Steve Jobs

**Now let's build it.**
