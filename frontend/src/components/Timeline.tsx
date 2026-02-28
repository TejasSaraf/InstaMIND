import type { Incident, SystemStatus } from '../types'

type TimelineProps = {
  incidents: Incident[]
  selectedId: string | null
  onSelect: (id: string) => void
}

const STATUS_STYLES: Record<SystemStatus, string> = {
  Normal: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40',
  Monitoring: 'bg-amber-500/20 text-amber-400 border-amber-500/40',
  ALERT: 'bg-red-500/20 text-red-400 border-red-500/40',
}

export function Timeline({ incidents, selectedId, onSelect }: TimelineProps) {
  return (
    <div className="rounded-xl border border-gray-800 bg-gray-800/50 shadow-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-700/50 bg-gray-800/80">
        <h2 className="text-sm font-semibold text-white flex items-center gap-2">
          <span>📊</span> Incident Timeline
        </h2>
      </div>
      <ul className="divide-y divide-gray-700/50 max-h-[280px] overflow-y-auto">
        {incidents.map((incident) => (
          <li key={incident.id}>
            <button
              type="button"
              onClick={() => onSelect(incident.id)}
              className={`w-full px-4 py-3 flex items-center gap-4 text-left transition-all ${
                selectedId === incident.id
                  ? 'bg-blue-500/10 border-l-4 border-l-blue-500'
                  : 'hover:bg-gray-700/30 border-l-4 border-l-transparent'
              }`}
            >
              <span className="text-gray-500 font-mono text-sm shrink-0 w-20">
                {incident.timestamp}
              </span>
              <span className="text-white font-medium flex-1 truncate">
                {incident.eventLabel}
              </span>
              <span
                className={`px-2 py-0.5 rounded text-xs font-medium border shrink-0 ${STATUS_STYLES[incident.status]}`}
              >
                {incident.status}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}
