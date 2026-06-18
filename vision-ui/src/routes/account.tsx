import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { AppShell } from "@/components/app-shell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Copy, Eye, EyeOff, RefreshCw } from "lucide-react";

export const Route = createFileRoute("/account")({
  head: () => ({ meta: [{ title: "Account — Vision" }, { name: "description", content: "Manage your profile, API tokens, and account settings." }] }),
  component: Account,
});

function Account() {
  const [showToken, setShowToken] = useState(false);
  const token = "vsn_live_8c4f1a92_d7e3b094a216f5dc7e34a8920bcd";

  return (
    <AppShell
      eyebrow="Chapter VII · Identity"
      title={<>Your <em className="italic text-cobalt-500">credentials,</em> kept tidy.</>}
      breadcrumb={["Settings", "Account"]}
    >
      <div className="grid grid-cols-12 gap-8">
        <div className="col-span-12 lg:col-span-8 space-y-10">
          <section className="paper-card rounded-lg p-8">
            <div className="flex items-start justify-between mb-6">
              <div>
                <div className="eyebrow">§ 01 · Profile</div>
                <h3 className="font-display text-3xl italic mt-2 leading-none">Alex Rivera</h3>
                <p className="font-mono text-[10px] uppercase tracking-widest text-ink-soft mt-2">Joined June 2, 2026 · Pro plan</p>
              </div>
              <div className="h-20 w-20 rounded-full bg-ink text-paper grid place-items-center font-display text-3xl italic">AR</div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mt-6 hairline pt-6">
              <div>
                <label className="eyebrow">Display name</label>
                <Input defaultValue="Alex Rivera" className="mt-2 bg-paper-tint border-rule" />
              </div>
              <div>
                <label className="eyebrow">Email</label>
                <Input defaultValue="alex.rivera@acme.com" disabled className="mt-2 bg-paper-tint border-rule" />
              </div>
              <div>
                <label className="eyebrow">Tenant ID</label>
                <div className="flex gap-1.5 mt-2">
                  <Input value="tnt_acme_8h2k9d" readOnly className="font-mono text-xs bg-paper-tint border-rule" />
                  <Button variant="outline" size="icon"><Copy className="h-3.5 w-3.5" /></Button>
                </div>
              </div>
              <div>
                <label className="eyebrow">Role</label>
                <Input value="Workspace owner" disabled className="mt-2 bg-paper-tint border-rule" />
              </div>
            </div>
            <div className="flex justify-end mt-6"><Button className="bg-ink text-paper hover:bg-cobalt-700 font-mono text-[10px] uppercase tracking-widest">Save profile</Button></div>
          </section>

          <section className="paper-card rounded-lg p-8">
            <div className="flex items-baseline justify-between">
              <div>
                <div className="eyebrow">§ 02 · API token</div>
                <h3 className="font-display text-3xl italic mt-2 leading-none">Programmatic access</h3>
              </div>
              <span className="font-mono text-[10px] uppercase tracking-widest text-ink-soft">Expires Sep 14, 2026</span>
            </div>
            <div className="mt-6 hairline pt-6 flex gap-1.5">
              <Input value={showToken ? token : "vsn_live_••••••••••••••••••••••••"} readOnly className="font-mono text-xs bg-paper-tint border-rule" />
              <Button variant="outline" size="icon" onClick={() => setShowToken(!showToken)}>{showToken ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}</Button>
              <Button variant="outline" size="icon"><Copy className="h-3.5 w-3.5" /></Button>
            </div>
            <div className="flex justify-between items-center mt-5">
              <p className="text-xs text-ink-soft">Regenerating invalidates the current token immediately.</p>
              <Button variant="outline" className="font-mono text-[10px] uppercase tracking-widest"><RefreshCw className="h-3 w-3 mr-1" /> Regenerate</Button>
            </div>
          </section>

          <section className="paper-card rounded-lg p-8 border-destructive/30 bg-destructive/5">
            <div className="eyebrow text-destructive">§ 03 · Danger zone</div>
            <h3 className="font-display text-3xl italic mt-2 text-destructive leading-none">Close this account</h3>
            <p className="text-sm text-ink-soft mt-3 max-w-md leading-relaxed">Deleting your account removes every document and query permanently. There is no recovery.</p>
            <Button variant="destructive" className="mt-5 font-mono text-[10px] uppercase tracking-widest">Delete account</Button>
          </section>
        </div>

        <aside className="col-span-12 lg:col-span-4 space-y-6">
          <div className="paper-card rounded-lg p-6">
            <div className="eyebrow">This month</div>
            <h4 className="font-display text-2xl italic mt-1">Usage</h4>
            <div className="mt-5 space-y-5">
              {[
                { l: "Queries", v: 128, max: 1000, u: "" },
                { l: "Tokens", v: 184, max: 2000, u: "k" },
                { l: "Storage", v: 42, max: 5000, u: " MB" },
              ].map((s) => (
                <div key={s.l}>
                  <div className="flex items-baseline justify-between font-mono text-[10px] uppercase tracking-widest text-ink-soft">
                    <span>{s.l}</span>
                    <span><span className="text-ink font-medium">{s.v}{s.u}</span> / {s.max}{s.u}</span>
                  </div>
                  <div className="h-px bg-rule mt-2 relative overflow-hidden">
                    <div className="absolute inset-y-0 left-0 bg-cobalt-500" style={{ width: `${(s.v / s.max) * 100}%`, height: 2 }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-ink text-paper rounded-lg p-6 grain relative">
            <div className="eyebrow text-paper/60 relative">Subscription</div>
            <div className="font-display text-5xl italic mt-3 text-cobalt-300 relative">Pro</div>
            <p className="font-mono text-[10px] uppercase tracking-widest text-paper/60 mt-2 relative">Renews Aug 2, 2026 · $79/mo</p>
            <button className="mt-6 font-mono text-[10px] uppercase tracking-widest text-paper border-b border-paper/40 pb-1 hover:border-cobalt-300 hover:text-cobalt-300 relative">
              Manage subscription →
            </button>
          </div>
        </aside>
      </div>
    </AppShell>
  );
}
