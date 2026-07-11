"use client";

import { useState, useEffect } from "react";
import { AppShell } from "@/components/layout/app-shell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { login, signup, logout, getCurrentUser, deleteAllData } from "@/lib/api";

export default function AccountPage() {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [deleting, setDeleting] = useState(false);
  const [deleteSuccess, setDeleteSuccess] = useState(false);

  const [loggedIn, setLoggedIn] = useState(false);
  const [tenant, setTenant] = useState("");
  const [userEmail, setUserEmail] = useState("");

  useEffect(() => {
    async function checkAuth() {
      try {
        const user = await getCurrentUser();
        if (user && user.tenant_id) {
          setLoggedIn(true);
          setTenant(user.tenant_id);
          setUserEmail(user.email);
        }
      } catch (e) {
        // Not logged in
        setLoggedIn(false);
      }
    }
    checkAuth();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      if (isLogin) {
        const res = await login({ email, password });
        setTenant(res.tenant_id || "tnt_unknown");
        setUserEmail(res.email || email);
      } else {
        const res = await signup({ email, password });
        setTenant(res.tenant_id || "tnt_unknown");
        setUserEmail(res.email || email);
      }
      setLoggedIn(true);
    } catch (err: any) {
      setError(err.message || "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      await logout();
    } catch (e) {
      console.error(e);
    }
    setLoggedIn(false);
    setTenant("");
    setUserEmail("");
  };

  const handleDeleteData = async () => {
    if (!window.confirm("Are you sure you want to delete all uploaded PDFs and embeddings? This action cannot be undone.")) return;
    setDeleting(true);
    setDeleteSuccess(false);
    setError("");
    try {
      await deleteAllData();
      setDeleteSuccess(true);
    } catch (e: any) {
      setError(e.message || "Failed to delete data");
    } finally {
      setDeleting(false);
    }
  };

  return (
    <AppShell eyebrow="Account" title="Your Settings" breadcrumb={["Workspace", "Account"]}>
      <div className="grid grid-cols-12 gap-8 mt-4">
        <div className="col-span-12 lg:col-span-5">
          {loggedIn ? (
            <div className="paper-card rounded-lg p-10 animate-fade-in">
              <span className="eyebrow">Active Session</span>
              <h3 className="font-display text-3xl italic mt-2">Welcome back.</h3>
              
              <div className="mt-8 space-y-4">
                <div>
                  <div className="eyebrow mb-1">Email</div>
                  <div className="font-mono text-sm">{userEmail || "Unknown"}</div>
                </div>
                <div>
                  <div className="eyebrow mb-1">Tenant ID</div>
                  <div className="font-mono text-sm text-cobalt-600">{tenant}</div>
                </div>
              </div>

              <div className="mt-10 hairline pt-6">
                <Button onClick={handleLogout} variant="outline" className="font-mono text-[10px] uppercase tracking-widest text-ink hover:bg-paper-tint">
                  Log out
                </Button>
              </div>

              <div className="mt-10 hairline pt-6 border-destructive/20">
                <span className="eyebrow text-destructive mb-2 block">Danger Zone</span>
                <p className="text-xs text-ink-soft mb-4">
                  Permanently delete all your uploaded PDFs, indexed embeddings, and metadata from the server. This action cannot be undone.
                </p>
                <Button onClick={handleDeleteData} disabled={deleting} variant="outline" className="font-mono text-[10px] uppercase tracking-widest text-destructive border-destructive/50 hover:bg-destructive hover:text-white transition-colors">
                  {deleting ? "Deleting..." : "Delete All My Data"}
                </Button>
                {deleteSuccess && <div className="mt-3 text-emerald-600 font-mono text-xs bg-emerald-50 p-2 rounded">All data deleted successfully.</div>}
                {error && <div className="mt-3 text-destructive font-mono text-xs bg-destructive/10 p-2 rounded">{error}</div>}
              </div>
            </div>
          ) : (
            <div className="paper-card rounded-lg p-10 animate-fade-in">
              <span className="eyebrow">Authentication</span>
              <h3 className="font-display text-3xl italic mt-2">{isLogin ? "Sign in to Vision." : "Create an account."}</h3>
              
              <form onSubmit={handleSubmit} className="mt-8 space-y-5">
                <div>
                  <Label className="eyebrow">Email address</Label>
                  <Input 
                    type="email" 
                    required 
                    value={email} 
                    onChange={e => setEmail(e.target.value)} 
                    className="mt-2 bg-paper-tint border-rule" 
                    placeholder="name@example.com"
                  />
                </div>
                <div>
                  <Label className="eyebrow">Password</Label>
                  <Input 
                    type="password" 
                    required 
                    value={password} 
                    onChange={e => setPassword(e.target.value)} 
                    className="mt-2 bg-paper-tint border-rule" 
                  />
                </div>
                
                {error && <div className="text-destructive font-mono text-xs bg-destructive/10 p-2 rounded">{error}</div>}

                <div className="pt-2">
                  <Button type="submit" disabled={loading} className="w-full bg-ink text-paper hover:bg-cobalt-700 font-mono text-[10px] uppercase tracking-widest">
                    {loading ? "Authenticating..." : (isLogin ? "Sign In" : "Sign Up")}
                  </Button>
                </div>
              </form>

              <div className="mt-6 text-center hairline pt-6">
                <button 
                  onClick={() => setIsLogin(!isLogin)} 
                  className="font-mono text-[10px] uppercase tracking-widest text-ink-soft hover:text-ink ink-link"
                >
                  {isLogin ? "Need an account? Sign up" : "Already have an account? Sign in"}
                </button>
              </div>
            </div>
          )}
        </div>

        <div className="col-span-12 lg:col-span-7 bg-ink text-paper p-10 lg:p-14 rounded-lg grain relative overflow-hidden flex flex-col justify-center">
           <h3 className="font-display text-4xl lg:text-5xl mt-4 leading-tight max-w-md relative">
            <em className="italic text-cobalt-300">Tenant-isolated</em> by design.
          </h3>
          <p className="mt-6 text-paper/70 leading-relaxed max-w-md font-mono text-xs">
            Every authenticated request is pinned to your tenant ID. You cannot access documents or queries from other tenants. Anonymous requests fall back to the default tenant.
          </p>
        </div>
      </div>
    </AppShell>
  );
}
