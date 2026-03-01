import { useRef } from 'react'

type InputBoxProps = {
  onFileSelect: (file: File | null) => void
  onSubmit: () => void
  processing: boolean
  drag: boolean
  onDragOver: (e: React.DragEvent) => void
  onDragLeave: (e: React.DragEvent) => void
  onDrop: (e: React.DragEvent) => void
  selectedFileName: string | null
}

function IconUpload({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
    </svg>
  )
}

function IconAnalyze({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
    </svg>
  )
}

export function InputBox({
  onFileSelect,
  onSubmit,
  processing,
  drag,
  onDragOver,
  onDragLeave,
  onDrop,
  selectedFileName,
}: InputBoxProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    onFileSelect(file || null)
    e.target.value = ''
  }

  return (
    <div className="rounded-2xl border border-neutral-800 bg-neutral-950 shadow-xl overflow-hidden">
      <div className="p-5 space-y-4">
        <div
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          className={`rounded-xl border-2 border-dashed p-6 text-center transition-colors ${
            drag
              ? 'border-teal-500 bg-teal-500/5'
              : 'border-neutral-700 hover:border-neutral-600 bg-neutral-900/50'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            className="hidden"
            aria-label="Upload video"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={processing}
            className="inline-flex items-center gap-2 text-sm font-medium text-neutral-300 hover:text-white hover:underline disabled:opacity-50"
          >
            <IconUpload className="w-4 h-4 shrink-0" />
            {selectedFileName ? selectedFileName : 'Upload video'}
          </button>
          <p className="text-xs text-neutral-500 mt-1">
            or drag and drop
          </p>
        </div>

        <button
          type="button"
          onClick={onSubmit}
          disabled={processing}
          className="w-full py-3 rounded-xl bg-teal-600 hover:bg-teal-500 disabled:bg-neutral-800 disabled:text-neutral-500 text-white font-medium transition-colors disabled:cursor-not-allowed border border-teal-500/30 flex items-center justify-center gap-2"
        >
          <IconAnalyze className="w-4 h-4 shrink-0" />
          {processing ? 'Processing...' : 'Analyze video'}
        </button>
      </div>
    </div>
  )
}
