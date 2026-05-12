import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Landingpage from "./components/Landingpage";
import { Navbar } from "./components/Navbar";
import { Dashboard } from "./pages/Dashboard";
import { Realtime } from "./pages/Realtime";
import { UploadVideo } from "./pages/UploadVideo";

const API_BASE = (
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.DEV ? "http://localhost:8000" : "")
).replace(/\/$/, "");

type AuthUser = {
  id: string;
  google_sub: string;
  email: string;
  name: string;
  picture?: string | null;
  email_verified: boolean;
  created_at: string;
  updated_at: string;
  last_login_at: string;
};

type AppView = "dashboard" | "upload" | "realtime";

const USER_STORAGE_KEY = "instamind.auth.user";

const _w = window as unknown as { __gsiInit?: boolean };
const gsi = {
  get initialized() {
    return !!_w.__gsiInit;
  },
  set initialized(v: boolean) {
    _w.__gsiInit = v;
  },
};

const VIEW_TO_PATH: Record<AppView, string> = {
  dashboard: "/dashboard",
  upload: "/upload",
  realtime: "/realtime",
};

function getViewFromPath(pathname: string): AppView {
  if (pathname === "/upload") return "upload";
  if (pathname === "/realtime") return "realtime";
  return "dashboard";
}

export default function App() {
  const [view, setView] = useState<AppView>(() =>
    getViewFromPath(window.location.pathname)
  );
  const [authUser, setAuthUser] = useState<AuthUser | null>(() => {
    const raw = localStorage.getItem(USER_STORAGE_KEY);
    if (!raw) return null;
    try {
      return JSON.parse(raw) as AuthUser;
    } catch {
      return null;
    }
  });
  const [authError, setAuthError] = useState<string | null>(null);

  const didValidate = useRef(false);
  useEffect(() => {
    if (didValidate.current) return;
    didValidate.current = true;
    const stored = authUser;
    if (!stored?.id) return;

    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 8_000);

    (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/auth/users/${stored.id}`, {
          signal: ctrl.signal,
        });
        const data = await res.json();
        if (!res.ok || !data.logged_in || !data.user) {
          setAuthUser(null);
          localStorage.removeItem(USER_STORAGE_KEY);
          return;
        }
        const dbUser = data.user as AuthUser;
        setAuthUser(dbUser);
        localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(dbUser));
      } catch {
      } finally {
        clearTimeout(timer);
      }
    })();

    return () => {
      ctrl.abort();
      clearTimeout(timer);
    };
  }, []);

  const handleGoogleCredential = useCallback(async (credential: string) => {
    setAuthError(null);
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 8_000);
    try {
      const res = await fetch(`${API_BASE}/api/v1/auth/google`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id_token: credential }),
        signal: ctrl.signal,
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Google sign-in failed.");
      }
      const user = data.user as AuthUser;
      setAuthUser(user);
      localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(user));
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        throw new Error("Sign-in timed out. Please try again.");
      }
      if (err instanceof TypeError) {
        throw new Error(
          "Cannot reach the backend server. Make sure it is running at " +
            API_BASE
        );
      }
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }, []);

  useEffect(() => {
    if (authUser) return;
    const clientId = import.meta.env.VITE_GOOGLE_CLIENT_ID;
    if (!clientId) return;

    let cancelled = false;

    const tryInit = () => {
      const google = window.google;
      if (!google || cancelled || gsi.initialized) return gsi.initialized;
      google.accounts.id.initialize({
        client_id: clientId,
        use_fedcm_for_prompt: false,
        callback: async (response: { credential?: string }) => {
          try {
            const token = response.credential;
            if (!token) {
              setAuthError("Google sign-in did not return a credential.");
              return;
            }
            await handleGoogleCredential(token);
          } catch (err) {
            const message =
              err instanceof Error ? err.message : "Google sign-in failed.";
            setAuthError(message);
          }
        },
      });
      gsi.initialized = true;
      return true;
    };

    if (!tryInit()) {
      const interval = setInterval(() => {
        if (tryInit()) clearInterval(interval);
      }, 250);

      const timeout = setTimeout(() => clearInterval(interval), 15_000);
      return () => {
        cancelled = true;
        clearInterval(interval);
        clearTimeout(timeout);
      };
    }
    return () => {
      cancelled = true;
    };
  }, [authUser, handleGoogleCredential]);

  const googleButtonRef = useCallback((element: HTMLDivElement | null) => {
    if (!element) return;

    const tryRender = () => {
      const google = window.google;
      if (!google || !gsi.initialized) return false;
      element.innerHTML = "";
      google.accounts.id.renderButton(element, {
        type: "standard",
        theme: "outline",
        size: "large",
        text: "signup_with",
        shape: "rectangular",
        width: 280,
      });
      return true;
    };

    if (!tryRender()) {
      const interval = setInterval(() => {
        if (tryRender()) clearInterval(interval);
      }, 250);
      setTimeout(() => clearInterval(interval), 10000);
    }
  }, []);

  const handleSignOut = useCallback(() => {
    setAuthError(null);
    setAuthUser(null);
    localStorage.removeItem(USER_STORAGE_KEY);
    setView("dashboard");
  }, []);

  const handleChangeView = useCallback((nextView: AppView) => {
    setView(nextView);
    const nextPath = VIEW_TO_PATH[nextView];
    if (window.location.pathname !== nextPath) {
      window.history.pushState(null, "", nextPath);
    }
  }, []);

  useEffect(() => {
    const pathname = window.location.pathname;

    if (!authUser) {
      if (pathname !== "/") {
        window.history.replaceState(null, "", "/");
      }
      return;
    }

    const nextView = getViewFromPath(pathname);
    const nextPath = VIEW_TO_PATH[nextView];

    if (pathname !== nextPath) {
      window.history.replaceState(null, "", nextPath);
    }

    if (view !== nextView) {
      setView(nextView);
    }
  }, [authUser, view]);

  useEffect(() => {
    const handlePopState = () => {
      const pathname = window.location.pathname;

      if (!authUser) {
        if (pathname !== "/") {
          window.history.replaceState(null, "", "/");
        }
        return;
      }

      setView(getViewFromPath(pathname));
    };

    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, [authUser]);

  const authName = useMemo(
    () => authUser?.name || authUser?.email || null,
    [authUser]
  );
  const authInitials = useMemo(() => {
    if (!authName) return null;
    const clean = authName.trim();
    if (!clean) return null;
    const parts = clean.split(/\s+/).filter(Boolean);
    if (parts.length >= 2) {
      return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
    }
    return clean.slice(0, 2).toUpperCase();
  }, [authName]);

  if (!authUser) {
    return (
      <Landingpage authError={authError} googleButtonRef={googleButtonRef} />
    );
  }

  return (
    <div className="min-h-screen h-screen flex flex-col bg-black text-white font-sans antialiased">
      <Navbar
        activeView={view}
        onChangeView={handleChangeView}
        authInitials={authInitials}
        onSignOut={handleSignOut}
      />
      <main className="flex-1 min-h-0 py-6 overflow-y-auto">
        {view === "dashboard" && <Dashboard />}
        {view === "upload" && <UploadVideo />}
        {view === "realtime" && <Realtime />}
      </main>
    </div>
  );
}
