function IconCheck({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
    </svg>
  )
}

function IconX({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
  )
}

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
          ? 'bg-neutral-900/90 border-neutral-600 text-neutral-200'
          : 'bg-neutral-900/90 border-neutral-600 text-neutral-200'
      }`}
    >
      {isSuccess ? (
        <IconCheck className="w-5 h-5 shrink-0 text-teal-400" />
      ) : (
        <IconX className="w-5 h-5 shrink-0 text-red-400" />
      )}
      <p className="text-sm font-medium flex-1">{message}</p>
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          className="p-1 rounded-lg hover:bg-neutral-700 transition-colors"
          aria-label="Dismiss"
        >
          <IconX className="w-4 h-4 text-neutral-400" />
        </button>
      )}
    </div>
  )
}
