type AlertBannerProps = {
  visible: boolean
}

export function AlertBanner({ visible }: AlertBannerProps) {
  if (!visible) return null

  return (
    <div
      className="flex items-center justify-center gap-2 py-3 px-4 bg-red-950/90 border-b border-red-500/50 animate-pulse"
      role="alert"
    >
      <span className="text-red-400 font-semibold text-sm sm:text-base">
        🚨 Emergency Detected
      </span>
    </div>
  )
}
