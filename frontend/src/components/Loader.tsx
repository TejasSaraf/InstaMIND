type LoaderProps = {
  message?: string
}

export function Loader({ message = 'Processing...' }: LoaderProps) {
  return (
    <div className="flex flex-col items-center justify-center py-12 gap-4">
      <div
        className="w-10 h-10 rounded-full border-2 border-neutral-700 border-t-teal-400 animate-spin"
        role="status"
        aria-label="Loading"
      />
      <p className="text-sm text-neutral-400">{message}</p>
    </div>
  )
}
