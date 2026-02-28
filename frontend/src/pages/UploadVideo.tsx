import { useState, useCallback, useEffect, useRef } from 'react'
import { Hero } from '../components/Hero'
import { InputBox } from '../components/InputBox'
import { Loader } from '../components/Loader'
import { ResultCard, type ResultData } from '../components/ResultCard'
import { Toast } from '../components/Toast'
import { HistorySidebar, type HistoryItem } from '../components/HistorySidebar'
import { HowItWorks } from '../components/HowItWorks'

const API_BASE = ''

type Video = {
  id: number
  originalName: string
  url: string
  mimeType: string | null
  size: number
  createdAt: string
}

function generatePlaceholderResult(videoName: string): ResultData {
  return {
    videoName,
    summary: `This video "${videoName}" has been processed. A full summary and key insights will appear here once analysis is connected. For now, this is a placeholder to show the result layout.`,
    insights: [
      'Video uploaded and stored successfully.',
      'Metadata saved for future analysis.',
      'You can revisit this result from the history sidebar.',
    ],
  }
}

export function UploadVideo() {
  const mainRef = useRef<HTMLDivElement>(null)
  const [videos, setVideos] = useState<Video[]>([])
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [processing, setProcessing] = useState(false)
  const [drag, setDrag] = useState(false)
  const [toast, setToast] = useState<{ type: 'success' | 'error'; message: string } | null>(null)
  const [currentResult, setCurrentResult] = useState<ResultData | null>(null)
  const [history, setHistory] = useState<HistoryItem[]>([])

  const fetchVideos = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/videos`)
      if (!res.ok) throw new Error('Failed to load videos')
      const data = await res.json()
      setVideos(data)
    } catch {
      setToast({ type: 'error', message: 'Could not load videos' })
    }
  }, [])

  useEffect(() => {
    fetchVideos()
  }, [fetchVideos])

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
    form.append('video', file)
    try {
      const res = await fetch(`${API_BASE}/api/videos`, { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Upload failed')

      setVideos((prev) => [data, ...prev])
      setSelectedFile(null)

      await new Promise((r) => setTimeout(r, 1500))
      const result = generatePlaceholderResult(file.name)
      setCurrentResult(result)

      const historyItem: HistoryItem = {
        id: `result-${Date.now()}`,
        videoName: file.name,
        createdAt: new Date().toISOString(),
        result,
      }
      setHistory((prev) => [historyItem, ...prev])
      setToast({ type: 'success', message: 'Video uploaded and processed' })
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

      {videos.length > 0 && (
        <section className="mt-12">
          <h2 className="text-lg font-semibold text-white mb-4">
            Your videos
          </h2>
          <ul className="space-y-3">
            {videos.slice(0, 5).map((v) => (
              <li
                key={v.id}
                className="rounded-xl border border-blue-800/50 bg-blue-950/40 p-4 flex flex-col sm:flex-row sm:items-center gap-3 shadow-lg shadow-blue-950/10"
              >
                <video
                  src={v.url}
                  controls
                  className="w-full sm:w-40 h-24 object-cover rounded-lg bg-blue-900"
                  preload="metadata"
                />
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-blue-100 truncate">
                    {v.originalName}
                  </p>
                  <p className="text-blue-200/60 text-sm">
                    {new Date(v.createdAt).toLocaleString()}
                  </p>
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  )
}
