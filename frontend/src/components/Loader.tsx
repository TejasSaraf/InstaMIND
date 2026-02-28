type LoaderProps = {
  message?: string
}

export function Loader({ message = 'Processing...' }: LoaderProps) {
  return (
    <div className="flex flex-col items-center justify-center py-12 gap-4">
      <div
        className="w-10 h-10 rounded-full border-2 border-blue-800/50 border-t-blue-400 animate-spin"
        role="status"
        aria-label="Loading"
      />
      <p className="text-sm text-blue-200/80">{message}</p>
    </div>
  )
}
