import type { Reasoning } from '../types'

type ReasonPanelProps = {
  data: Reasoning | null
}

export function ReasonPanel({ data }: ReasonPanelProps) {
  if (!data) {
    return (
      <div className="rounded-xl border border-gray-800 bg-gray-800/50 shadow-xl overflow-hidden h-full min-h-[320px] flex items-center justify-center">
        <p className="text-gray-500 text-sm">Select an incident from the timeline</p>
      </div>
    )
  }

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-800/50 shadow-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-700/50 bg-gray-800/80">
        <h2 className="text-sm font-semibold text-white flex items-center gap-2">
          <span>🧠</span> AI Reasoning
        </h2>
      </div>
      <div className="p-4 space-y-4">
        <div>
          <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Event type</p>
          <p className="text-white font-medium capitalize">{data.eventName}</p>
        </div>

        <div>
          <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Confidence</p>
          <div className="flex items-center gap-2">
            <div className="flex-1 h-2.5 rounded-full bg-gray-700 overflow-hidden">
              <div
                className="h-full rounded-full bg-blue-500 transition-all duration-500"
                style={{ width: `${data.confidence * 100}%` }}
              />
            </div>
            <span className="text-sm text-gray-300 font-medium tabular-nums w-10">
              {(data.confidence * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        <div>
          <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">Reason</p>
          <ul className="space-y-1.5">
            {data.reasons.map((reason, i) => (
              <li key={i} className="flex gap-2 text-sm text-gray-400">
                <span className="text-blue-400 shrink-0">•</span>
                <span className="leading-snug">{reason}</span>
              </li>
            ))}
          </ul>
        </div>

        <div>
          <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Decision</p>
          <span
            className={`inline-flex px-3 py-1.5 rounded-lg text-sm font-semibold ${
              data.decision === 'ALERT'
                ? 'bg-red-500/20 text-red-400 border border-red-500/50'
                : 'bg-amber-500/20 text-amber-400 border border-amber-500/50'
            }`}
          >
            {data.decision}
          </span>
        </div>
      </div>
    </div>
  )
}
