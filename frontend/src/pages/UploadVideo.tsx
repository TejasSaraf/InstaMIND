import { useState, useCallback, useEffect, useRef } from 'react'
import { Hero } from '../components/Hero'
import { InputBox } from '../components/InputBox'
import { Loader } from '../components/Loader'
import { ResultCard, type ResultData } from '../components/ResultCard'
import { Toast } from '../components/Toast'
import { HistorySidebar, type HistoryItem } from '../components/HistorySidebar'
import { HowItWorks } from '../components/HowItWorks'

const API_BASE = (
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.DEV ? 'http://localhost:8000' : '')
).replace(/\/$/, '')

type ApiIncident = {
  incident_type: string
  confidence: number
  timestamp_seconds: number
  evidence: string
  recommended_action: string
}

type ApiReport = {
  report_id: string
  source_filename: string
  created_at: string
  processing_time_ms: number
  met_latency_target: boolean
  summary: string
  incidents: ApiIncident[]
  raw_signals?: {
    latency?: {
      p95_ms?: number
      max_ms?: number
      violations?: number
    }
  }
}

function reportToResult(report: ApiReport): ResultData {
  const incidentLines = report.incidents
    .filter((x) => x.incident_type !== 'none')
    .slice(0, 4)
    .map(
      (x) =>
        `${x.incident_type} (${Math.round(x.confidence * 100)}%) at ${x.timestamp_seconds.toFixed(1)}s: ${x.evidence}`,
    )

  const latency = report.raw_signals?.latency
  const latencyLine = latency
    ? `Frame latency p95 ${Number(latency.p95_ms || 0).toFixed(1)}ms, max ${Number(latency.max_ms || 0).toFixed(1)}ms, violations ${latency.violations || 0}.`
    : `Upload analysis latency ${report.processing_time_ms.toFixed(1)}ms.`

  return {
    videoName: report.source_filename,
    summary: report.summary,
    insights: [
      latencyLine,
      `Latency target met: ${report.met_latency_target ? 'Yes' : 'No'}.`,
      ...(incidentLines.length > 0 ? incidentLines : ['No critical incidents detected.']),
    ],
  }
}

export function UploadVideo() {
  const mainRef = useRef<HTMLDivElement>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [processing, setProcessing] = useState(false)
  const [drag, setDrag] = useState(false)
  const [toast, setToast] = useState<{ type: 'success' | 'error'; message: string } | null>(null)
  const [currentResult, setCurrentResult] = useState<ResultData | null>(null)
  const [history, setHistory] = useState<HistoryItem[]>([])

  const fetchReports = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/reports`)
      if (!res.ok) throw new Error('Failed to load reports')
      const data = (await res.json()) as { reports?: ApiReport[] }
      const reports = data.reports || []
      const mapped: HistoryItem[] = reports.map((report) => ({
        id: report.report_id,
        videoName: report.source_filename,
        createdAt: report.created_at,
        result: reportToResult(report),
      }))
      setHistory(mapped)
    } catch {
      setToast({
        type: 'error',
        message: 'Could not load reports. Is the backend running? Start it with: cd backend && python -m uvicorn app.main:app --port 8000',
      })
    }
  }, [])

  useEffect(() => {
    fetchReports()
  }, [fetchReports])

  const scrollToInput = () => {
    mainRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const handleFileSelect = useCallback((file: File | null) => {
    setSelectedFile(file)
  }, [])

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDrag(false)
    const file = e.dataTransfer.files[0]
    if (file?.type.startsWith('video/')) {
      setSelectedFile(file)
    }
  }, [])

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDrag(true)
  }, [])

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDrag(false)
  }, [])

  const submit = useCallback(async () => {
    const file = selectedFile
    if (!file || !file.type.startsWith('video/')) {
      setToast({ type: 'error', message: 'Please select a video file first' })
      return
    }

    setProcessing(true)
    setToast(null)
    setCurrentResult(null)

    const form = new FormData()
    form.append('file', file)
    try {
      const res = await fetch(`${API_BASE}/api/v1/analyze/upload`, { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || data.error || 'Upload failed')

      const report = data.report as ApiReport
      const result = reportToResult(report)
      setSelectedFile(null)
      setCurrentResult(result)

      const historyItem: HistoryItem = {
        id: report.report_id,
        videoName: report.source_filename,
        createdAt: report.created_at,
        result,
      }
      setHistory((prev) => [historyItem, ...prev])
      setToast({ type: 'success', message: 'Video analyzed successfully' })
    } catch (e) {
      setToast({ type: 'error', message: e instanceof Error ? e.message : 'Upload failed' })
    } finally {
      setProcessing(false)
    }
  }, [selectedFile])

  const handleHistorySelect = useCallback((item: HistoryItem) => {
    if (item.result) setCurrentResult(item.result)
  }, [])

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 pb-16">
      <div className="py-8">
        <Hero onGetStarted={scrollToInput} />
      </div>

      <div ref={mainRef} className="grid lg:grid-cols-[1fr,280px] gap-8 items-start">
        <div className="space-y-6 min-w-0">
          <InputBox
            onFileSelect={handleFileSelect}
            onSubmit={submit}
            processing={processing}
            drag={drag}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            selectedFileName={selectedFile?.name ?? null}
          />

          {toast && (
            <Toast
              type={toast.type}
              message={toast.message}
              onDismiss={() => setToast(null)}
            />
          )}

          {processing && <Loader message="Generating response..." />}

          {!processing && currentResult && (
            <ResultCard data={currentResult} />
          )}
        </div>

        <div className="lg:sticky lg:top-24">
          <HistorySidebar items={history} onSelect={handleHistorySelect} />
        </div>
      </div>

      <div className="mt-12">
        <HowItWorks />
      </div>
    </div>
  )
}
