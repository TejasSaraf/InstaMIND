import { useCallback, useEffect, useRef, useState } from "react";

type VideoFile = {
  name: string;
  path: string;
  category: string;
};

type BBox = { x: number; y: number; w: number; h: number; confidence: number };

const API_BASE = (
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.DEV ? "http://localhost:8000" : "")
).replace(/\/$/, "");

const DETECT_INTERVAL_MS = 1500;
const LOAD_STAGGER_MS = 250;
const DETECT_STAGGER_MS = 150;
const CAPTURE_MAX_WIDTH = 480;
const LAZY_ROOT_MARGIN = "400px";
const EMPTY_HOLD_TICKS = 3;

const VIDEO_FILES: VideoFile[] = [
  {
    name: "Normal_Videos_478_x264.mp4",
    path: "/data-videos/Normal_Videos_478_x264.mp4",
    category: "Normal",
  },
  {
    name: "Robbery008_x264.mp4",
    path: "/data-videos/Robbery008_x264.mp4",
    category: "Robbery",
  },
  {
    name: "Robbery047_x264.mp4",
    path: "/data-videos/Robbery047_x264.mp4",
    category: "Robbery",
  },
  {
    name: "Shooting006_x264.mp4",
    path: "/data-videos/Shooting006_x264.mp4",
    category: "Shooting",
  },
  {
    name: "Shooting023_x264.mp4",
    path: "/data-videos/Shooting023_x264.mp4",
    category: "Shooting",
  },
  {
    name: "Shooting030_x264.mp4",
    path: "/data-videos/Shooting030_x264.mp4",
    category: "Shooting",
  },
  {
    name: "Shoplifting032_x264.mp4",
    path: "/data-videos/Shoplifting032_x264.mp4",
    category: "Shoplifting",
  },
  {
    name: "Shoplifting039_x264.mp4",
    path: "/data-videos/Shoplifting039_x264.mp4",
    category: "Shoplifting",
  },
  {
    name: "Shoplifting044_x264.mp4",
    path: "/data-videos/Shoplifting044_x264.mp4",
    category: "Shoplifting",
  },
  {
    name: "Fighting036_x264.mp4",
    path: "/data-videos/Fighting036_x264.mp4",
    category: "Fighting",
  },
  {
    name: "Fighting040_x264.mp4",
    path: "/data-videos/Fighting040_x264.mp4",
    category: "Fighting",
  },
  {
    name: "Fighting042_x264.mp4",
    path: "/data-videos/Fighting042_x264.mp4",
    category: "Fighting",
  },
];

type IncidentSeverity = "critical" | "high" | "moderate";

type Incident = {
  id: string;
  cameraIndex: number;
  cameraId: string;
  type: string;
  severity: IncidentSeverity;
  confidence: number;
  detectedAt: number;
  alerted: boolean;
};

const SEVERITY_BY_CATEGORY: Record<string, IncidentSeverity> = {
  Robbery: "critical",
  Shooting: "critical",
  Fighting: "high",
  Shoplifting: "moderate",
};

const SEVERITY_STYLES: Record<IncidentSeverity, string> = {
  critical: "text-orange-300 border-orange-700/70 bg-orange-950/40",
  high: "text-amber-300 border-amber-800/60 bg-amber-950/40",
  moderate: "text-yellow-300 border-yellow-800/60 bg-yellow-950/40",
};

function buildInitialIncidents(): Incident[] {
  const now = Date.now();
  const out: Incident[] = [];
  VIDEO_FILES.forEach((v, i) => {
    if (v.category === "Normal") return;
    const minutesAgo = (i + 1) * 2;
    const conf = 0.68 + ((i * 7) % 28) / 100;
    out.push({
      id: `inc-${i}`,
      cameraIndex: i,
      cameraId: `CAM-${String(i + 1).padStart(2, "0")}`,
      type: v.category,
      severity: SEVERITY_BY_CATEGORY[v.category] ?? "moderate",
      confidence: Math.min(0.96, conf),
      detectedAt: now - minutesAgo * 60_000,
      alerted: false,
    });
  });
  return out;
}

function timeAgo(ms: number): string {
  const diff = Math.max(0, Date.now() - ms);
  const m = Math.floor(diff / 60_000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  return `${h}h ago`;
}

export function Dashboard() {
  const totalCameras = VIDEO_FILES.length;
  const [onlineStates, setOnlineStates] = useState<boolean[]>(() =>
    new Array(totalCameras).fill(false)
  );
  const [incidents, setIncidents] = useState<Incident[]>(() =>
    buildInitialIncidents()
  );

  const setCardOnline = useCallback(
    (cameraIndex: number, isOnline: boolean) => {
      setOnlineStates((prev) => {
        if (prev[cameraIndex] === isOnline) return prev;
        const next = prev.slice();
        next[cameraIndex] = isOnline;
        return next;
      });
    },
    []
  );

  const dismissIncident = useCallback((id: string) => {
    setIncidents((prev) => prev.filter((inc) => inc.id !== id));
  }, []);

  const alertIncident = useCallback((id: string) => {
    setIncidents((prev) =>
      prev.map((inc) => (inc.id === id ? { ...inc, alerted: true } : inc))
    );
  }, []);

  const onlineCount = onlineStates.reduce((n, v) => n + (v ? 1 : 0), 0);
  const allOnline = onlineCount === totalCameras;
  const anyOnline = onlineCount > 0;

  return (
    <div className="relative max-w-[1800px] mx-auto px-4 sm:px-6 lg:pl-[18rem]">
      <aside className="hidden lg:block fixed left-4 top-24 w-[15rem] h-[calc(100vh-7rem)] overflow-y-auto scrollbar-hide">
        <div className="mb-6">
          <p className="text-xs text-neutral-500 uppercase tracking-wide mb-1">
            Online Cameras
          </p>
          <p className="text-2xl font-bold text-white">
            {onlineCount}
            <span className="text-base text-neutral-500 font-medium">
              {" "}
              / {totalCameras}
            </span>
          </p>
          <div className="mt-2 flex items-center gap-2">
            <span
              className={`w-2 h-2 rounded-full ${
                allOnline
                  ? "bg-emerald-400 animate-pulse"
                  : anyOnline
                  ? "bg-amber-400 animate-pulse"
                  : "bg-neutral-600"
              }`}
            />
            <span
              className={`text-xs ${
                allOnline
                  ? "text-emerald-400"
                  : anyOnline
                  ? "text-amber-400"
                  : "text-neutral-500"
              }`}
            >
              {allOnline
                ? "All feeds active"
                : anyOnline
                ? `${totalCameras - onlineCount} connecting…`
                : "No feeds online"}
            </span>
          </div>
        </div>

        <div className="border-t border-neutral-800 pt-4">
          <div className="flex items-center justify-between mb-3">
            <p className="text-xs text-neutral-500 uppercase tracking-wide">
              Recent Incidents
            </p>
            <span className="text-[10px] font-mono text-neutral-500">
              {incidents.length}
            </span>
          </div>
          {incidents.length === 0 ? (
            <p className="text-[11px] text-neutral-500 italic">
              All clear. No active incidents.
            </p>
          ) : (
            <ul className="space-y-2">
              {incidents.map((inc) => (
                <IncidentItem
                  key={inc.id}
                  incident={inc}
                  onDismiss={() => dismissIncident(inc.id)}
                  onAlert={() => alertIncident(inc.id)}
                />
              ))}
            </ul>
          )}
        </div>
      </aside>

      <div
        aria-hidden
        className="hidden lg:block fixed top-24 left-[16.5rem] h-[calc(100vh-7rem)] w-px bg-neutral-800"
      />

      <div className="py-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
          {VIDEO_FILES.map((video, i) => (
            <VideoCard
              key={video.name}
              video={video}
              index={i}
              onOnlineChange={setCardOnline}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function VideoCard({
  video,
  index,
  onOnlineChange,
}: {
  video: VideoFile;
  index: number;
  onOnlineChange: (cameraIndex: number, isOnline: boolean) => void;
}) {
  const cardRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const inFlightRef = useRef(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const missCountRef = useRef(0);

  const [inView, setInView] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [ready, setReady] = useState(false);
  const [detections, setDetections] = useState<BBox[]>([]);

  useEffect(() => {
    onOnlineChange(index, ready);
    return () => onOnlineChange(index, false);
  }, [index, ready, onOnlineChange]);

  useEffect(() => {
    const node = cardRef.current;
    if (!node) return;
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setInView(true);
          obs.disconnect();
        }
      },
      { rootMargin: LAZY_ROOT_MARGIN }
    );
    obs.observe(node);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    if (!inView || loaded) return;
    const t = setTimeout(() => setLoaded(true), index * LOAD_STAGGER_MS);
    return () => clearTimeout(t);
  }, [inView, loaded, index]);

  useEffect(() => {
    if (!ready) return;

    const captureFrameB64 = (): string | null => {
      const v = videoRef.current;
      if (!v || v.paused || v.ended || !v.videoWidth) return null;
      const off = document.createElement("canvas");
      off.width = Math.min(v.videoWidth, CAPTURE_MAX_WIDTH);
      off.height = Math.round(v.videoHeight * (off.width / v.videoWidth));
      const ctx = off.getContext("2d");
      if (!ctx) return null;
      ctx.drawImage(v, 0, 0, off.width, off.height);
      return off.toDataURL("image/jpeg", 0.6).split(",")[1];
    };

    const postDetect = async () => {
      if (inFlightRef.current) return;
      const frame = captureFrameB64();
      if (!frame) return;
      inFlightRef.current = true;
      try {
        const res = await fetch(`${API_BASE}/api/v1/detect/frame`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ frame }),
        });
        if (!res.ok) return;
        const data = await res.json();
        const next: BBox[] = data.detections || [];
        if (next.length > 0) {
          missCountRef.current = 0;
          setDetections(next);
        } else {
          missCountRef.current += 1;
          if (missCountRef.current >= EMPTY_HOLD_TICKS) setDetections([]);
        }
      } catch {
      } finally {
        inFlightRef.current = false;
      }
    };

    startTimeoutRef.current = setTimeout(() => {
      postDetect();
      intervalRef.current = setInterval(postDetect, DETECT_INTERVAL_MS);
    }, index * DETECT_STAGGER_MS);

    return () => {
      if (startTimeoutRef.current) clearTimeout(startTimeoutRef.current);
      if (intervalRef.current) clearInterval(intervalRef.current);
      inFlightRef.current = false;
    };
  }, [ready, index]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const v = videoRef.current;
    if (!canvas || !v) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = v.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (detections.length === 0 || !v.videoWidth) return;

    const captureW = Math.min(v.videoWidth, CAPTURE_MAX_WIDTH);
    const captureH = Math.round(v.videoHeight * (captureW / v.videoWidth));
    const sx = rect.width / captureW;
    const sy = rect.height / captureH;

    ctx.strokeStyle = "#00FF00";
    ctx.lineWidth = 2;

    for (const d of detections) {
      ctx.strokeRect(d.x * sx, d.y * sy, d.w * sx, d.h * sy);
    }
  }, [detections]);

  return (
    <article
      ref={cardRef}
      className="rounded-lg border border-neutral-800 bg-black/50 overflow-hidden"
    >
      <div className="relative aspect-video bg-neutral-900">
        {loaded && (
          <video
            ref={videoRef}
            className="w-full h-full object-cover"
            src={video.path}
            autoPlay
            loop
            muted
            playsInline
            preload="metadata"
            onPlaying={() => setReady(true)}
            onCanPlay={() => setReady(true)}
            onLoadedData={() => setReady(true)}
            onWaiting={() => setReady(false)}
            onStalled={() => setReady(false)}
            onEmptied={() => setReady(false)}
            onError={() => setReady(false)}
          />
        )}

        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />

        {}
        {!ready && (
          <div className="absolute inset-0 flex items-center justify-center bg-neutral-950/80">
            <div className="flex flex-col items-center gap-2">
              <div className="w-6 h-6 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin" />
              <span className="text-[10px] uppercase tracking-wider text-neutral-500 font-mono">
                {loaded ? "Buffering…" : "Queued"}
              </span>
            </div>
          </div>
        )}

        <div className="absolute top-2 left-2 flex gap-2">
          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-black/70 border border-white/10 text-neutral-300">
            CAM-{String(index + 1).padStart(2, "0")}
          </span>
        </div>

        <div
          className={`absolute top-2 right-2 flex items-center gap-1.5 px-1.5 py-0.5 rounded bg-black/70 border border-white/10 ${
            ready ? "text-emerald-400" : "text-neutral-500"
          }`}
        >
          <WifiIcon className="w-3.5 h-3.5" />
          <BatteryIcon className="w-4 h-4" />
        </div>
      </div>
    </article>
  );
}

function IncidentItem({
  incident,
  onDismiss,
  onAlert,
}: {
  incident: Incident;
  onDismiss: () => void;
  onAlert: () => void;
}) {
  const sevClass = SEVERITY_STYLES[incident.severity];
  return (
    <li className="rounded-md border border-neutral-800 bg-neutral-900/50 p-2">
      <div className="flex items-center justify-between gap-2 mb-1">
        <span className="text-[10px] font-mono text-neutral-400">
          {incident.cameraId}
        </span>
        <span
          className={`text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded border ${sevClass}`}
        >
          {incident.severity}
        </span>
      </div>

      <p className="text-xs font-medium text-white leading-tight">
        {incident.type}
      </p>
      <p className="text-[10px] text-neutral-500 mt-0.5">
        {timeAgo(incident.detectedAt)} · {Math.round(incident.confidence * 100)}
        %
      </p>

      <div className="mt-2 flex gap-1.5">
        <button
          type="button"
          onClick={onDismiss}
          className="flex-1 text-[10px] uppercase tracking-wider font-medium px-2 py-1 rounded border border-neutral-700 text-neutral-300 hover:bg-neutral-800 hover:text-white transition"
        >
          Dismiss
        </button>
        <button
          type="button"
          onClick={onAlert}
          disabled={incident.alerted}
          className={`flex-1 text-[10px] uppercase tracking-wider font-medium px-2 py-1 rounded border transition ${
            incident.alerted
              ? "border-emerald-700 bg-emerald-950/50 text-emerald-300 cursor-default"
              : "border-orange-700/70 bg-orange-950/40 text-orange-200 hover:bg-orange-900/50 hover:text-orange-100"
          }`}
        >
          {incident.alerted ? "Alerted" : "Alert"}
        </button>
      </div>
    </li>
  );
}

function WifiIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M5 12.55a11 11 0 0 1 14 0" />
      <path d="M1.42 9a16 16 0 0 1 21.16 0" />
      <path d="M8.53 16.11a6 6 0 0 1 6.95 0" />
      <line x1="12" y1="20" x2="12.01" y2="20" />
    </svg>
  );
}

function BatteryIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <rect x="2" y="7" width="16" height="10" rx="2" ry="2" />
      <line x1="22" y1="11" x2="22" y2="13" />
      {}
      <rect
        x="4"
        y="9"
        width="12"
        height="6"
        rx="0.5"
        fill="currentColor"
        stroke="none"
      />
    </svg>
  );
}
