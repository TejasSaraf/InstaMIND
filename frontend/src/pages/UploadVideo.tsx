import { useState, useCallback, useEffect, useRef } from "react";
import { Hero } from "../components/Hero";
import { InputBox } from "../components/InputBox";
import { type ResultData } from "../components/ResultCard";
import { Toast } from "../components/Toast";

const API_BASE = (
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.DEV ? "http://localhost:8000" : "")
).replace(/\/$/, "");

const FRAME_INTERVAL_MS = 3000;

type PlaybackMarker = {
  t: number;
  incidentType: string;
  confidence: number;
  note?: string;
};

type BBox = { x: number; y: number; w: number; h: number; confidence: number };

export function UploadVideo() {
  const mainRef = useRef<HTMLDivElement>(null);
  const previewVideoRef = useRef<HTMLVideoElement>(null);
  const detectCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const captureIntervalRef = useRef<ReturnType<typeof setInterval> | null>(
    null
  );
  const detectIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const inFlightRef = useRef(false);
  const detectInFlightRef = useRef(false);
  const monitorRunningRef = useRef(false);

  const [drag, setDrag] = useState(false);
  const [toast, setToast] = useState<{
    type: "success" | "error";
    message: string;
  } | null>(null);
  const [currentResult, setCurrentResult] = useState<ResultData | null>(null);
  const [previewVideoUrl, setPreviewVideoUrl] = useState<string | null>(null);
  const [previewFileName, setPreviewFileName] = useState<string | null>(null);
  const [monitoringPlayback, setMonitoringPlayback] = useState(false);
  const [liveMoments, setLiveMoments] = useState<
    Array<{ time: string; text: string; confidence: number }>
  >([]);
  const [playbackMarkers, setPlaybackMarkers] = useState<PlaybackMarker[]>([]);
  const [playbackHud, setPlaybackHud] = useState<{
    timeLabel: string;
    incidentLabel: string;
  }>({
    timeLabel: "00:00",
    incidentLabel: "Press play to begin live incident analysis",
  });
  const [summaryText, setSummaryText] = useState<string>(
    "Waiting for analysis to begin..."
  );
  const [detections, setDetections] = useState<BBox[]>([]);
  const currentVideoUrlRef = useRef<string | null>(null);

  useEffect(() => {
    return () => {
      stopFrameCapture();
      if (currentVideoUrlRef.current)
        URL.revokeObjectURL(currentVideoUrlRef.current);
    };
  }, []);

  const handleFileSelect = useCallback((file: File | null) => {
    stopFrameCapture();
    setLiveMoments([]);
    setPlaybackMarkers([]);
    setCurrentResult(null);
    setPlaybackHud({
      timeLabel: "00:00",
      incidentLabel: "Press play to begin live incident analysis",
    });

    if (currentVideoUrlRef.current) {
      URL.revokeObjectURL(currentVideoUrlRef.current);
      currentVideoUrlRef.current = null;
    }

    if (file?.type.startsWith("video/")) {
      const url = URL.createObjectURL(file);
      currentVideoUrlRef.current = url;
      setPreviewVideoUrl(url);
      setPreviewFileName(file.name);
    } else {
      setPreviewVideoUrl(null);
      setPreviewFileName(null);
    }
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDrag(false);
      const file = e.dataTransfer.files[0];
      if (file?.type.startsWith("video/")) handleFileSelect(file);
    },
    [handleFileSelect]
  );

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDrag(true);
  }, []);
  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDrag(false);
  }, []);

  const captureFrameB64 = useCallback((): string | null => {
    const video = previewVideoRef.current;
    if (!video || video.paused || video.ended || !video.videoWidth) return null;
    const canvas = document.createElement("canvas");
    canvas.width = Math.min(video.videoWidth, 896);
    canvas.height = Math.round(
      video.videoHeight * (canvas.width / video.videoWidth)
    );
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
    return dataUrl.split(",")[1];
  }, []);

  const postFrame = useCallback(
    async (frameB64: string, videoSeconds: number) => {
      if (inFlightRef.current) return;
      inFlightRef.current = true;
      try {
        const res = await fetch(`${API_BASE}/api/v1/analyze/frame`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ frame: frameB64, timestamp: videoSeconds }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Frame analysis failed");

        const incidentType = data.incident_type || "normal";
        const confidence = data.confidence || 0;
        const evidence = data.evidence || "";

        const mm = String(Math.floor(videoSeconds / 60)).padStart(2, "0");
        const ss = String(Math.floor(videoSeconds % 60)).padStart(2, "0");
        const label =
          incidentType === "none" || incidentType === "normal"
            ? "No critical event detected."
            : `Detected ${incidentType.replaceAll("_", " ")}`;

        setLiveMoments((prev) =>
          [...prev, { time: `${mm}:${ss}`, text: label, confidence }].slice(-18)
        );

        if (evidence) setSummaryText(evidence);

        if (incidentType !== "none" && incidentType !== "normal") {
          const tEnd = Math.max(0, videoSeconds);
          const key = `${incidentType}-${tEnd.toFixed(1)}`;
          setPlaybackMarkers((prev) => {
            if (prev.some((m) => `${m.incidentType}-${m.t.toFixed(1)}` === key))
              return prev;
            return [
              ...prev,
              { t: tEnd, incidentType, confidence, note: evidence },
            ].sort((a, b) => a.t - b.t);
          });
        }

        setCurrentResult({
          videoName: previewFileName || "video",
          summary: evidence || "Analysis complete.",
          keyMoments: [],
          insights: [
            incidentType !== "normal"
              ? `${incidentType
                  .replace(/_/g, " ")
                  .replace(/\b\w/g, (c: string) =>
                    c.toUpperCase()
                  )} detected with ${Math.round(confidence * 100)}% confidence.`
              : "No critical incidents detected.",
            data.recommended_action && data.recommended_action !== "none"
              ? `Recommended action: ${data.recommended_action}`
              : "",
          ].filter(Boolean),
          videoUrl: previewVideoUrl || currentVideoUrlRef.current || undefined,
        });
      } catch (err) {
        setToast({
          type: "error",
          message: err instanceof Error ? err.message : "Frame analysis failed",
        });
      } finally {
        inFlightRef.current = false;
      }
    },
    [previewVideoUrl, previewFileName]
  );

  const EMPTY_HOLD_TICKS = 3;
  const missCountRef = useRef(0);

  const postDetect = useCallback(async (frameB64: string) => {
    if (detectInFlightRef.current) return;
    detectInFlightRef.current = true;
    try {
      const res = await fetch(`${API_BASE}/api/v1/detect/frame`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frame: frameB64 }),
      });
      if (!res.ok) return;
      const data = await res.json();
      const next: BBox[] = data.detections || [];
      if (next.length > 0) {
        missCountRef.current = 0;
        setDetections(next);
      } else {
        missCountRef.current += 1;
        if (missCountRef.current >= EMPTY_HOLD_TICKS) {
          setDetections([]);
        }
      }
    } catch {
    } finally {
      detectInFlightRef.current = false;
    }
  }, []);

  useEffect(() => {
    const canvas = detectCanvasRef.current;
    const video = previewVideoRef.current;
    if (!canvas || !video) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (detections.length === 0 || !video.videoWidth) return;

    const captureW = Math.min(video.videoWidth, 896);
    const captureH = Math.round(
      video.videoHeight * (captureW / video.videoWidth)
    );
    const sx = rect.width / captureW;
    const sy = rect.height / captureH;

    ctx.strokeStyle = "#00FF00";
    ctx.lineWidth = 2;

    for (const d of detections) {
      ctx.strokeRect(d.x * sx, d.y * sy, d.w * sx, d.h * sy);
    }
  }, [detections]);

  const stopFrameCapture = useCallback(() => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    if (detectIntervalRef.current) {
      clearInterval(detectIntervalRef.current);
      detectIntervalRef.current = null;
    }
    monitorRunningRef.current = false;
    inFlightRef.current = false;
    detectInFlightRef.current = false;
    setMonitoringPlayback(false);
    setDetections([]);
  }, []);

  const startFrameCapture = useCallback(() => {
    const video = previewVideoRef.current;
    if (!video || monitorRunningRef.current) return;

    monitorRunningRef.current = true;
    setMonitoringPlayback(true);
    setDetections([]);
    setToast(null);

    const frame = captureFrameB64();
    if (frame) {
      postFrame(frame, video.currentTime);
      postDetect(frame);
    }

    captureIntervalRef.current = setInterval(() => {
      if (!monitorRunningRef.current) return;
      const f = captureFrameB64();
      const t = previewVideoRef.current?.currentTime ?? 0;
      if (f) postFrame(f, t);
    }, FRAME_INTERVAL_MS);

    detectIntervalRef.current = setInterval(() => {
      if (!monitorRunningRef.current) return;
      const f = captureFrameB64();
      if (f) postDetect(f);
    }, 500);
  }, [captureFrameB64, postFrame, postDetect]);

  const onPlaybackTimeUpdate = useCallback(() => {
    const v = previewVideoRef.current;
    if (!v) return;
    const t = v.currentTime;
    const mm = String(Math.floor(t / 60)).padStart(2, "0");
    const ss = String(Math.floor(t % 60)).padStart(2, "0");
    const frac = String(Math.floor((t % 1) * 100)).padStart(2, "0");
    const timeLabel = `${mm}:${ss}.${frac}`;

    const active = pickActiveMarker(playbackMarkers, t);
    let incidentLabel: string;
    if (!active) {
      incidentLabel =
        playbackMarkers.length === 0
          ? `Press play — Gemma 3 analyzes a frame every ${
              FRAME_INTERVAL_MS / 1000
            }s on-device`
          : "No flagged incident at this point";
    } else if (active.incidentType === "none") {
      incidentLabel = "No critical event at this timestamp";
    } else {
      incidentLabel = `${active.incidentType.replaceAll(
        "_",
        " "
      )} — ${Math.round(active.confidence * 100)}% confidence`;
    }

    setPlaybackHud({ timeLabel, incidentLabel });
  }, [playbackMarkers]);

  const handleAnalyzeClick = useCallback(() => {
    if (!previewVideoRef.current) return;
    previewVideoRef.current.play();
    startFrameCapture();
  }, [startFrameCapture]);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 pb-16">
      <div className="">
        <Hero />
      </div>

      <div
        ref={mainRef}
        className="grid lg:grid-cols-[2fr_3fr] gap-6 items-start"
      >
        {}

        <div className="space-y-6 min-w-0">
          {}
          {previewVideoUrl && (
            <div className="rounded-xl overflow-hidden">
              <div className="space-y-2">
                {}
                <div className="relative rounded-xl overflow-hidden border border-neutral-800 bg-black">
                  <video
                    ref={previewVideoRef}
                    className="w-full h-full object-cover aspect-video bg-black"
                    controls
                    src={previewVideoUrl}
                    onEnded={stopFrameCapture}
                    onTimeUpdate={onPlaybackTimeUpdate}
                    onSeeked={onPlaybackTimeUpdate}
                  />

                  {}
                  <canvas
                    ref={detectCanvasRef}
                    className="absolute inset-0 w-full h-full pointer-events-none"
                  />

                  {}
                  <div
                    className="pointer-events-none absolute left-0 right-0 bottom-12 px-3 flex justify-center"
                    aria-live="polite"
                  >
                    <div className="max-w-[92%] rounded-lg border border-white/10 bg-black/80 px-3 py-2 text-xs sm:text-sm text-white shadow-lg backdrop-blur-sm">
                      <div className="font-mono tabular-nums text-amber-300/95">
                        {playbackHud.timeLabel}
                      </div>
                      <div className="mt-0.5 text-neutral-100 leading-snug">
                        {playbackHud.incidentLabel}
                      </div>
                    </div>
                  </div>
                </div>

                <button
                  type="button"
                  onClick={
                    monitoringPlayback ? stopFrameCapture : handleAnalyzeClick
                  }
                  className="w-full text-sm font-semibold text-white border-1 border-gray-600
                   rounded-lg px-4 py-2.5 transition hover:border-gray-200 cursor-pointer"
                >
                  {monitoringPlayback ? "Stop Analysis" : "Analyze Video"}
                </button>
              </div>
            </div>
          )}

          {}
          <InputBox
            onFileSelect={handleFileSelect}
            drag={drag}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            selectedFileName={previewFileName}
          />

          {toast && (
            <Toast
              type={toast.type}
              message={toast.message}
              onDismiss={() => setToast(null)}
            />
          )}
        </div>
        {}
        <div className="space-y-6">
          <div className="rounded-2xl border border-neutral-800 bg-neutral-950 shadow-xl overflow-hidden min-h-[320px] flex flex-col">
            <div className="px-5 py-2 border-b border-neutral-800 flex items-center justify-between">
              <h3 className="text-base font-semibold text-white">
                Scene Summary
              </h3>
              {monitoringPlayback && (
                <span className="text-[10px] uppercase tracking-wider font-bold text-purple-400 bg-purple-950/40 px-2 py-1 rounded border border-purple-800/50">
                  Gemma 3
                </span>
              )}
            </div>

            <div className="flex-1 p-5 overflow-y-auto">
              {monitoringPlayback ? (
                <div className="space-y-3">
                  <div className="flex gap-2 items-center mb-4">
                    <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                    <span className="text-xs text-neutral-400 font-mono">
                      LIVE ANALYSIS
                    </span>
                  </div>
                  <p className="text-sm leading-relaxed text-neutral-200">
                    {summaryText}
                  </p>
                </div>
              ) : currentResult ? (
                <div className="space-y-5">
                  <section>
                    <h4 className="text-xs font-semibold uppercase tracking-wider text-neutral-500 mb-2">
                      Summary
                    </h4>
                    <p className="text-sm text-neutral-300 leading-relaxed">
                      {currentResult.summary}
                    </p>
                  </section>
                  {currentResult.insights.length > 0 && (
                    <section>
                      <h4 className="text-xs font-semibold uppercase tracking-wider text-neutral-500 mb-2">
                        Key Insights
                      </h4>
                      <ul className="space-y-2">
                        {currentResult.insights.map((insight, i) => (
                          <li
                            key={i}
                            className="flex gap-2 text-sm text-neutral-300"
                          >
                            <span className="text-neutral-500 mt-0.5 shrink-0">
                              •
                            </span>
                            <span className="leading-relaxed">{insight}</span>
                          </li>
                        ))}
                      </ul>
                    </section>
                  )}
                  {currentResult.keyMoments &&
                    currentResult.keyMoments.length > 0 && (
                      <section>
                        <h4 className="text-xs font-semibold uppercase tracking-wider text-neutral-500 mb-2">
                          Key Moments
                        </h4>
                        <ul className="space-y-2">
                          {currentResult.keyMoments.map((m, i) => (
                            <li
                              key={i}
                              className="flex gap-3 text-sm text-neutral-300"
                            >
                              <span className="text-[11px] rounded border border-neutral-700 bg-neutral-900 px-2 py-0.5 text-neutral-200 shrink-0">
                                {String(
                                  Math.floor(m.timestampSeconds / 60)
                                ).padStart(2, "0")}
                                :
                                {String(m.timestampSeconds % 60).padStart(
                                  2,
                                  "0"
                                )}
                              </span>
                              <span className="leading-relaxed">{m.text}</span>
                            </li>
                          ))}
                        </ul>
                      </section>
                    )}
                </div>
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-center space-y-2 opacity-50">
                  <svg
                    className="w-8 h-8 text-neutral-500"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                    />
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                    />
                  </svg>
                  <p className="text-sm text-neutral-400">
                    Model inactive.
                    <br />
                    Upload and analyze to see insights.
                  </p>
                </div>
              )}
            </div>
          </div>

          {}
          {liveMoments.length > 0 && (
            <ul className="space-y-2 max-h-56 overflow-y-auto pr-1">
              {liveMoments.map((m, idx) => (
                <li
                  key={`${m.time}-${idx}`}
                  className="flex items-start gap-3 rounded-lg border border-neutral-800 bg-neutral-900/40 px-3 py-2 text-neutral-300"
                >
                  <span className="text-[11px] rounded border border-neutral-700 bg-neutral-900 px-2 py-0.5 text-neutral-200 shrink-0">
                    {m.time}
                  </span>
                  <div className="min-w-0">
                    <p className="leading-relaxed">{m.text}</p>
                    <p className="text-[11px] text-neutral-500 mt-1">
                      confidence {(m.confidence * 100).toFixed(0)}%
                    </p>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

function pickActiveMarker(
  markers: PlaybackMarker[],
  t: number
): PlaybackMarker | null {
  const eps = 0.35;
  let best: PlaybackMarker | null = null;
  for (const m of markers) {
    if (m.t <= t + eps) best = m;
  }
  return best;
}
