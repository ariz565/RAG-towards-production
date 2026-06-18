import { Link, useRouterState } from "@tanstack/react-router";
import { type ReactNode } from "react";
import { Search, Command } from "lucide-react";

const navItems = [
  { to: "/", num: "01", label: "Overview" },
  { to: "/documents", num: "02", label: "Documents" },
  { to: "/chat", num: "03", label: "Ask" },
  { to: "/summarize", num: "04", label: "Summarize" },
  { to: "/inspector", num: "05", label: "Inspector" },
  { to: "/config", num: "06", label: "Configuration" },
  { to: "/account", num: "07", label: "Account" },
] as const;

function VisionMark() {
  return (
    <Link to="/" className="group inline-flex items-center gap-2.5">
      <div className="relative h-9 w-9 rounded-md bg-ink grid place-items-center">
        <span className="font-display italic text-paper text-lg leading-none">V</span>
        <span className="absolute -bottom-1 -right-1 h-2 w-2 bg-cobalt-500 rounded-full" />
      </div>
      <div className="leading-tight">
        <div className="font-display text-[22px] italic tracking-tight text-ink">Vision</div>
        <div className="eyebrow -mt-0.5">Document Intelligence</div>
      </div>
    </Link>
  );
}

export function AppShell({
  children,
  title,
  eyebrow,
  breadcrumb,
}: {
  children: ReactNode;
  title?: ReactNode;
  eyebrow?: string;
  breadcrumb?: string[];
}) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const current = navItems.find((n) => (n.to === "/" ? pathname === "/" : pathname.startsWith(n.to)));

  return (
    <div className="min-h-screen bg-paper text-ink">
      <aside className="fixed inset-y-0 left-0 hidden lg:flex w-[260px] flex-col border-r border-rule bg-paper px-7 py-7 z-40">
        <VisionMark />

        <div className="mt-10">
          <div className="eyebrow mb-4">Workspace</div>
          <nav className="flex flex-col">
            {navItems.map((item) => {
              const active = current?.to === item.to;
              return (
                <Link
                  key={item.to}
                  to={item.to}
                  className={`group relative flex items-baseline gap-3 py-2.5 border-b border-rule last:border-b-0 transition-colors ${
                    active ? "text-ink" : "text-ink-soft hover:text-ink"
                  }`}
                >
                  <span className={`font-mono text-[10px] tracking-widest ${active ? "text-cobalt-500" : "text-ink-soft/60"}`}>{item.num}</span>
                  <span className="font-display text-lg leading-none">{item.label}</span>
                  {active && <span className="absolute right-0 top-1/2 -translate-y-1/2 h-1.5 w-1.5 rounded-full bg-cobalt-500" />}
                </Link>
              );
            })}
          </nav>
        </div>

        <div className="mt-auto pt-8">
          <div className="paper-card rounded-lg p-4 bg-paper-tint">
            <div className="eyebrow">Tenant</div>
            <div className="mt-1 font-display text-lg italic leading-tight">Acme Corp</div>
            <div className="mt-2 font-mono text-[10px] text-ink-soft">tnt_acme_8h2k9d · Pro</div>
          </div>
        </div>
      </aside>

      <div className="lg:pl-[260px]">
        <header className="sticky top-0 z-30 bg-paper/85 backdrop-blur-md border-b border-rule">
          <div className="flex items-center justify-between px-6 lg:px-12 h-14">
            <div className="flex items-center gap-2 text-xs font-mono text-ink-soft">
              {breadcrumb?.length ? breadcrumb.map((b, i) => (
                <span key={i} className="flex items-center gap-2">
                  {i > 0 && <span className="opacity-40">/</span>}
                  <span className={i === breadcrumb.length - 1 ? "text-ink" : ""}>{b}</span>
                </span>
              )) : (
                <span>{current?.num} — {current?.label}</span>
              )}
            </div>
            <div className="flex items-center gap-1.5">
              <button className="inline-flex items-center gap-2 text-xs font-mono text-ink-soft hover:text-ink border border-rule px-2.5 py-1.5 rounded-md bg-card">
                <Search className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Search</span>
                <span className="hidden sm:inline-flex items-center gap-0.5 ml-2 text-[10px] opacity-60"><Command className="h-3 w-3" />K</span>
              </button>
              <Link to="/account" className="ml-2 h-8 w-8 rounded-full bg-ink text-paper grid place-items-center font-display text-sm">AR</Link>
            </div>
          </div>

          {(title || eyebrow) && (
            <div className="px-6 lg:px-12 pb-8 pt-6">
              {eyebrow && <div className="eyebrow mb-3">{eyebrow}</div>}
              {title && <h1 className="serif-display text-[44px] lg:text-[56px] text-ink max-w-4xl">{title}</h1>}
            </div>
          )}
        </header>

        <main className="px-6 lg:px-12 py-10 animate-fade-in max-w-[1280px]">
          {children}
        </main>

        <footer className="px-6 lg:px-12 py-10 border-t border-rule mt-16 text-xs font-mono text-ink-soft flex items-center justify-between">
          <span>Vision · v1.4.2 · {new Date().getFullYear()}</span>
          <span className="inline-flex items-center gap-2"><span className="h-1.5 w-1.5 rounded-full bg-emerald-500" /> All systems operational</span>
        </footer>
      </div>
    </div>
  );
}
