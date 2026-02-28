type ToastProps = {
  type: 'success' | 'error'
  message: string
  onDismiss?: () => void
}

export function Toast({ type, message, onDismiss }: ToastProps) {
  const isSuccess = type === 'success'
  return (
    <div
      role="alert"
      className={`flex items-center gap-3 rounded-xl border px-4 py-3 shadow-lg ${
        isSuccess
          ? 'bg-blue-950/60 border-blue-600/50 text-blue-100'
          : 'bg-red-950/40 border-red-700/50 text-red-200'
      }`}
    >
      {isSuccess ? (
        <svg className="w-5 h-5 shrink-0 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
        </svg>
      ) : (
        <svg className="w-5 h-5 shrink-0 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
        </svg>
      )}
      <p className="text-sm font-medium flex-1">{message}</p>
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          className="p-1 rounded-lg hover:bg-white/10 transition-colors"
          aria-label="Dismiss"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      )}
    </div>
  )
}
