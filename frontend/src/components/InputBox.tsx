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
    <div className="rounded-2xl border border-blue-800/50 bg-blue-950/40 shadow-xl shadow-blue-950/20 overflow-hidden">
      <div className="p-5 space-y-4">
        <div
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          className={`rounded-xl border-2 border-dashed p-6 text-center transition-colors ${
            drag
              ? 'border-blue-500 bg-blue-500/10'
              : 'border-blue-800/60 hover:border-blue-600/80 bg-blue-950/30'
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
            className="text-sm font-medium text-blue-300 hover:text-blue-200 hover:underline disabled:opacity-50"
          >
            {selectedFileName ? `📎 ${selectedFileName}` : '📎 Upload video'}
          </button>
          <p className="text-xs text-blue-200/60 mt-1">
            or drag and drop
          </p>
        </div>

        <button
          type="button"
          onClick={onSubmit}
          disabled={processing}
          className="w-full py-3 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:bg-blue-900/80 disabled:text-blue-300/60 text-white font-medium transition-colors disabled:cursor-not-allowed shadow-lg shadow-blue-500/20"
        >
          {processing ? 'Processing...' : 'Analyze video'}
        </button>
      </div>
    </div>
  )
}
