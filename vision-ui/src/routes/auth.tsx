import { createFileRoute, useNavigate, Link } from "@tanstack/react-router";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ArrowRight } from "lucide-react";

export const Route = createFileRoute("/auth")({
  head: () => ({
    meta: [
      { title: "Enter Vision" },
      { name: "description", content: "Sign in or create your Vision account to start asking questions of your documents." },
    ],
  }),
  component: Auth,
});

function Auth() {
  const [mode, setMode] = useState<"signup" | "login">("signup");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setTimeout(() => { setLoading(false); navigate({ to: "/" }); }, 700);
  };

  return (
    <div className="min-h-screen bg-paper text-ink grid grid-cols-1 lg:grid-cols-12">
      <aside className="lg:col-span-7 relative bg-ink text-paper p-10 lg:p-16 flex flex-col justify-between grain overflow-hidden">
        <div className="absolute inset-0 opacity-[0.06] pointer-events-none animate-slow-pan"
          style={{ backgroundImage: "radial-gradient(circle at 20% 20%, #fff 0, transparent 40%), radial-gradient(circle at 80% 70%, #fff 0, transparent 35%)" }} />

        <header className="relative flex items-center justify-between">
          <Link to="/" className="inline-flex items-center gap-2.5">
            <div className="h-9 w-9 rounded-md bg-paper grid place-items-center"><span className="font-display italic text-ink text-lg leading-none">V</span></div>
            <span className="font-display text-2xl italic">Vision</span>
          </Link>
          <span className="font-mono text-[10px] tracking-widest text-paper/60">EST. 2026 · NEW YORK</span>
        </header>

        <div className="relative max-w-2xl">
          <span className="eyebrow text-paper/60">Volume I · Issue 01</span>
          <h1 className="serif-display text-[72px] lg:text-[112px] mt-6 leading-[0.92]">
            Read every<br/>
            <em className="italic text-cobalt-300">document</em><br/>
            as if you<br/>
            wrote it.
          </h1>
          <p className="mt-8 max-w-md text-paper/75 text-base leading-relaxed">
            Vision is a quiet, exact instrument for understanding contracts, policies, and reports. Upload a PDF, ask a question, receive an answer with every claim cited to a page.
          </p>
        </div>

        <footer className="relative flex items-end justify-between font-mono text-[10px] tracking-widest text-paper/50 uppercase">
          <div className="space-y-1">
            <div>Multi-tenant · Encrypted</div>
            <div>HR · Legal · Finance</div>
          </div>
          <div className="text-right">
            <div>Hybrid BM25 + Vector</div>
            <div>Any LLM, one API</div>
          </div>
        </footer>
      </aside>

      <section className="lg:col-span-5 flex items-center justify-center px-8 py-16 lg:p-16">
        <div className="w-full max-w-md">
          <div className="eyebrow">{mode === "signup" ? "New reader" : "Welcome back"}</div>
          <h2 className="serif-display text-5xl mt-3 leading-tight">
            {mode === "signup" ? <>Begin your <em className="italic text-cobalt-500">first chapter.</em></> : <>Sign in to <em className="italic text-cobalt-500">continue.</em></>}
          </h2>
          <p className="mt-4 text-ink-soft">
            {mode === "signup" ? "A workspace is ready for you in under a minute." : "We saved your place exactly where you left it."}
          </p>

          <form onSubmit={submit} className="mt-10 space-y-5">
            {mode === "signup" && (
              <div>
                <Label htmlFor="name" className="eyebrow">Your name</Label>
                <Input id="name" placeholder="Alex Rivera" required className="mt-2 h-12 bg-card border-rule rounded-md" />
              </div>
            )}
            <div>
              <Label htmlFor="email" className="eyebrow">Work email</Label>
              <Input id="email" type="email" placeholder="you@company.com" required className="mt-2 h-12 bg-card border-rule rounded-md" />
            </div>
            <div>
              <div className="flex items-center justify-between">
                <Label htmlFor="password" className="eyebrow">Password</Label>
                {mode === "login" && <a className="text-[11px] font-mono text-cobalt-500 hover:text-cobalt-700 cursor-pointer">forgot?</a>}
              </div>
              <Input id="password" type="password" placeholder="••••••••" minLength={8} required className="mt-2 h-12 bg-card border-rule rounded-md" />
            </div>
            <Button type="submit" size="lg" disabled={loading} className="w-full h-12 bg-ink text-paper hover:bg-cobalt-700 rounded-md font-mono text-xs uppercase tracking-widest group">
              {loading ? "Just a moment…" : (
                <>
                  {mode === "signup" ? "Create account" : "Enter Vision"}
                  <ArrowRight className="h-4 w-4 ml-2 group-hover:translate-x-1 transition-transform" />
                </>
              )}
            </Button>
          </form>

          <div className="mt-10 pt-8 border-t border-rule flex items-center justify-between text-sm">
            <span className="text-ink-soft">{mode === "signup" ? "Already a reader?" : "First time here?"}</span>
            <button onClick={() => setMode(mode === "signup" ? "login" : "signup")} className="font-display italic text-cobalt-500 hover:text-cobalt-700 text-base ink-link">
              {mode === "signup" ? "Sign in →" : "Create an account →"}
            </button>
          </div>

          <p className="mt-12 text-[11px] font-mono text-ink-soft text-center uppercase tracking-widest">
            By continuing you agree to our <a className="ink-link cursor-pointer">terms</a> & <a className="ink-link cursor-pointer">privacy</a>.
          </p>
        </div>
      </section>
    </div>
  );
}
