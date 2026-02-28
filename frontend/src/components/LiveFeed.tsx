import type { SystemStatus } from '../types'

type LiveFeedProps = {
  status: SystemStatus
}

export function LiveFeed({ status }: LiveFeedProps) {
  const isAlert = status === 'ALERT'

  return (
    <div
      className={`rounded-xl overflow-hidden shadow-xl transition-all duration-300 ${
        isAlert
          ? 'ring-2 ring-red-500/80 shadow-red-500/20 animate-pulse'
          : 'border border-gray-800 shadow-black/20'
      }`}
    >
      <div className="aspect-video bg-gray-900 flex items-center justify-center relative">
        <span className="text-gray-600 font-medium text-lg">Live Feed</span>
        {isAlert && (
          <div className="absolute inset-0 pointer-events-none ring-2 ring-red-500/30 rounded-xl" />
        )}
      </div>
      <div className="px-4 py-2 bg-gray-800/80 border-t border-gray-700/50 text-center text-xs text-gray-500">
        Camera 1 · Real-time stream
      </div>
    </div>
  )
}
