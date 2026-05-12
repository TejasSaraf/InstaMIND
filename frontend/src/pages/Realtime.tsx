import { useCallback, useEffect, useRef, useState } from "react";

const API_BASE = (
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.DEV ? "http://localhost:8000" : "")
).replace(/\/$/, "");

type KeyMoment = {
  elapsedSeconds: number;
  note: string;
  confidence: number;
  incidentType: string;
};

type BBox = { x: number; y: number; w: number; h: number; confidence: number };

const FRAME_INTERVAL_MS = 3000;
const DETECT_INTERVAL_MS = 500;

function incidentColor(type: string): string {
  if (type === "none")
    return "text-neutral-300 border-neutral-700 bg-neutral-900/60";
  if (["violent_activity", "shooting", "robbery"].includes(type))
    return "text-rose-300 border-rose-700 bg-rose-950/60";
  if (["fainting", "choking"].includes(type))
    return "text-amber-300 border-amber-700 bg-amber-950/60";
  return "text-yellow-300 border-yellow-700 bg-yellow-950/60";
}

function fmtElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60)
    .toString()
    .padStart(2, "0");
  const s = Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${m}:${s}`;
}

export function Realtime() {
  const remoteVideoRef = useRef<HTMLVideoElement | null>(null);
  const localStreamRef = useRef<MediaStream | null>(null);
  const pcSenderRef = useRef<RTCPeerConnection | null>(null);
  const pcReceiverRef = useRef<RTCPeerConnection | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const captureIntervalRef = useRef<ReturnType<typeof setInterval> | null>(
    null
  );
  const detectIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const detectInFlightRef = useRef(false);
  const inFlightRef = useRef(false);
  const analysisRunningRef = useRef(false);
  const streamStartRef = useRef<number>(0);

  const [connecting, setConnecting] = useState(true);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysisRunning, setAnalysisRunning] = useState(false);
  const [keyMoments, setKeyMoments] = useState<KeyMoment[]>([]);
  const [streamSeconds, setStreamSeconds] = useState(0);
  const [summaryText, setSummaryText] = useState<string>(
    "Waiting for analysis to begin..."
  );
  const [detections, setDetections] = useState<BBox[]>([]);

  const [hud, setHud] = useState<{
    incidentType: string;
    confidence: number;
    label: string;
  }>({
    incidentType: "none",
    confidence: 0,
    label: "Start analysis to detect incidents",
  });

  useEffect(() => {
    let mounted = true;
    initWebRTC();

    async function initWebRTC() {
      setConnecting(true);
      setError(null);
      try {
        const localStream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "environment",
            width: { ideal: 1920 },
            height: { ideal: 1080 },
            frameRate: { ideal: 30 },
          },
          audio: true,
        });
        if (!mounted) return;
        localStreamRef.current = localStream;

        const sender = new RTCPeerConnection();
        const receiver = new RTCPeerConnection();
        pcSenderRef.current = sender;
        pcReceiverRef.current = receiver;

        sender.onicecandidate = async (e) => {
          if (e.candidate) await receiver.addIceCandidate(e.candidate);
        };
        receiver.onicecandidate = async (e) => {
          if (e.candidate) await sender.addIceCandidate(e.candidate);
        };
        receiver.ontrack = (e) => {
          if (remoteVideoRef.current)
            remoteVideoRef.current.srcObject = e.streams[0];
        };

        localStream.getTracks().forEach((track) => {
          const rtpSender = sender.addTrack(track, localStream);
          if (track.kind === "video") {
            const params = rtpSender.getParameters();
            params.encodings = [
              { maxBitrate: 4_000_000, scaleResolutionDownBy: 1 },
            ];
            rtpSender.setParameters(params).catch(() => {});
          }
        });

        const offer = await sender.createOffer();
        await sender.setLocalDescription(offer);
        await receiver.setRemoteDescription(offer);
        const answer = await receiver.createAnswer();
        await receiver.setLocalDescription(answer);
        await sender.setRemoteDescription(answer);

        if (mounted) setConnected(true);
      } catch (err) {
        if (mounted) {
          setConnected(false);
          setError(
            err instanceof Error
              ? err.message
              : "Could not start camera. Allow camera access."
          );
        }
      } finally {
        if (mounted) setConnecting(false);
      }
    }

    return () => {
      mounted = false;
      stopAnalysis();
      pcSenderRef.current?.close();
      pcReceiverRef.current?.close();
      pcSenderRef.current = null;
      pcReceiverRef.current = null;
      localStreamRef.current?.getTracks().forEach((t) => t.stop());
      localStreamRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!analysisRunning) return;
    const timer = setInterval(() => {
      setStreamSeconds(
        Math.floor((Date.now() - streamStartRef.current) / 1000)
      );
    }, 500);
    return () => clearInterval(timer);
  }, [analysisRunning]);

  const captureFrameB64 = useCallback((): string | null => {
    const video = remoteVideoRef.current;
    if (!video || video.paused || !video.videoWidth) return null;
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
    const canvas = canvasRef.current;
    const video = remoteVideoRef.current;
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

  const postFrame = useCallback(async (frameB64: string, elapsed: number) => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    try {
      const res = await fetch(`${API_BASE}/api/v1/analyze/frame`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frame: frameB64, timestamp: elapsed }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Frame analysis failed");

      const incidentType = data.incident_type || "normal";
      const confidence = data.confidence || 0;
      const evidence = data.evidence || "";

      const note =
        incidentType === "none" || incidentType === "normal"
          ? "No critical event detected."
          : `Detected ${incidentType.replaceAll("_", " ")}`;

      setHud({
        incidentType,
        confidence,
        label:
          incidentType === "none" || incidentType === "normal"
            ? "No incident detected"
            : `${incidentType.replaceAll("_", " ")} — ${Math.round(
                confidence * 100
              )}% confidence`,
      });

      if (evidence) setSummaryText(evidence);

      setKeyMoments((prev) =>
        [
          ...prev,
          { elapsedSeconds: elapsed, note, confidence, incidentType },
        ].slice(-14)
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis error");
    } finally {
      inFlightRef.current = false;
    }
  }, []);

  useEffect(() => {
    if (!connected) return;

    const tick = () => {
      const f = captureFrameB64();
      if (f) postDetect(f);
    };
    tick();
    detectIntervalRef.current = setInterval(tick, DETECT_INTERVAL_MS);

    return () => {
      if (detectIntervalRef.current) {
        clearInterval(detectIntervalRef.current);
        detectIntervalRef.current = null;
      }
      detectInFlightRef.current = false;
      setDetections([]);
    };
  }, [connected, captureFrameB64, postDetect]);

  const startAnalysis = useCallback(() => {
    if (!connected || analysisRunning) return;

    streamStartRef.current = Date.now();
    setStreamSeconds(0);
    setAnalysisRunning(true);
    analysisRunningRef.current = true;
    setKeyMoments([]);
    setError(null);
    setHud({
      incidentType: "none",
      confidence: 0,
      label: "Analyzing first frame…",
    });

    const frame = captureFrameB64();
    if (frame) postFrame(frame, 0);

    captureIntervalRef.current = setInterval(() => {
      if (!analysisRunningRef.current) return;
      const f = captureFrameB64();
      const elapsed = Math.floor((Date.now() - streamStartRef.current) / 1000);
      if (f) postFrame(f, elapsed);
    }, FRAME_INTERVAL_MS);
  }, [connected, analysisRunning, captureFrameB64, postFrame]);

  const stopAnalysis = useCallback(() => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    inFlightRef.current = false;
    analysisRunningRef.current = false;
    setAnalysisRunning(false);
    setHud({ incidentType: "none", confidence: 0, label: "Analysis stopped" });
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 space-y-6">
      <section className="rounded-2xl bg-neutral-950 shadow-xl overflow-hidden">
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr] min-h-[560px]">
          {}
          <div className="flex flex-col">
            {}
            <div className="p-4 flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-white">Live Feed</h3>
                <span
                  className={`text-xs px-2 py-1 rounded border ${
                    connected
                      ? "text-emerald-300 border-emerald-700 bg-emerald-950/30"
                      : "text-neutral-300 border-neutral-700 bg-neutral-900/40"
                  }`}
                >
                  {connecting ? "Connecting…" : connected ? "Live" : "Offline"}
                </span>
              </div>

              {error && <p className="text-sm text-rose-300">{error}</p>}

              {}
              <div className="relative rounded-xl border border-neutral-800 bg-black overflow-hidden">
                <video
                  ref={remoteVideoRef}
                  autoPlay
                  muted
                  playsInline
                  className="w-full aspect-video bg-black object-cover"
                />

                {}
                <canvas
                  ref={canvasRef}
                  className="absolute inset-0 w-full h-full pointer-events-none"
                />

                {}
                {analysisRunning && (
                  <div className="absolute top-2 left-2 flex gap-2">
                    <div className="font-mono text-xs text-white bg-black/70 rounded px-2 py-0.5 border border-white/10">
                      {fmtElapsed(streamSeconds)}
                    </div>
                  </div>
                )}

                {}
                {analysisRunning && (
                  <div
                    className="pointer-events-none absolute left-0 right-0 bottom-3 px-3 flex justify-center"
                    aria-live="polite"
                  >
                    <div
                      className={`rounded-lg border px-3 py-1.5 text-xs text-white shadow-lg backdrop-blur-sm max-w-[92%] ${incidentColor(
                        hud.incidentType
                      )} bg-black/80`}
                    >
                      {hud.label}
                      {hud.incidentType !== "none" && hud.confidence > 0 && (
                        <span className="ml-2 opacity-70">
                          {Math.round(hud.confidence * 100)}%
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>

              <button
                type="button"
                onClick={analysisRunning ? stopAnalysis : startAnalysis}
                disabled={!connected}
                className="w-full text-sm text-white border border-neutral-700 rounded-lg px-3 py-2 transition hover:bg-neutral-800 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {analysisRunning ? "Stop Analysis" : "Start Analysis"}
              </button>
            </div>
          </div>

          {}
          <div className="p-5 flex flex-col gap-5">
            {}
            <div className="flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-white">
                  Scene Summary
                </h3>
                {analysisRunning && (
                  <span className="text-[10px] uppercase tracking-wider font-bold text-purple-400 bg-purple-950/40 px-2 py-1 rounded border border-purple-800/50">
                    Gemma 3
                  </span>
                )}
              </div>
              <div className="rounded-xl bg-neutral-900/50 border border-neutral-800/80 p-4">
                {analysisRunning ? (
                  <div className="space-y-3">
                    <div className="flex gap-2 items-center mb-2">
                      <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                      <span className="text-xs text-neutral-400 font-mono">
                        LIVE ANALYSIS
                      </span>
                    </div>
                    <p className="text-sm leading-relaxed text-neutral-200">
                      {summaryText}
                    </p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center text-center space-y-2 opacity-50 py-6">
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
                      Start analysis to see live insights.
                    </p>
                  </div>
                )}
              </div>
            </div>

            {}
            <div className="flex flex-col gap-3 flex-1">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-semibold text-white">
                  Key Moments
                </h4>
                {analysisRunning && (
                  <span className="text-xs text-emerald-400 animate-pulse">
                    Live — every {FRAME_INTERVAL_MS / 1000}s
                  </span>
                )}
                {!analysisRunning && keyMoments.length > 0 && (
                  <span className="text-xs text-neutral-400">
                    Analysis stopped
                  </span>
                )}
              </div>
              {keyMoments.length === 0 ? (
                <p className="text-sm text-neutral-500">
                  Start analysis to generate model-detected key moments with
                  stream timestamps.
                </p>
              ) : (
                <ul className="space-y-2 max-h-[300px] overflow-y-auto pr-1">
                  {keyMoments.map((moment, index) => (
                    <li
                      key={`${moment.elapsedSeconds}-${index}`}
                      className="flex items-start gap-3 rounded-lg border border-neutral-800 bg-neutral-900/40 px-3 py-2 text-neutral-300"
                    >
                      <span
                        className={`text-[11px] rounded border px-2 py-0.5 shrink-0 font-mono ${
                          moment.incidentType === "none"
                            ? "text-neutral-400 border-neutral-700 bg-neutral-900"
                            : "text-amber-200 border-amber-800 bg-amber-950/40"
                        }`}
                      >
                        {fmtElapsed(moment.elapsedSeconds)}
                      </span>
                      <div className="min-w-0">
                        <p className="leading-relaxed text-sm">{moment.note}</p>
                        {moment.incidentType !== "none" && (
                          <p className="text-[11px] text-neutral-500 mt-0.5">
                            confidence {(moment.confidence * 100).toFixed(0)}%
                          </p>
                        )}
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
