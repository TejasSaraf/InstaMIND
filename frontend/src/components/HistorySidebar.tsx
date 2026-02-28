import type { ResultData } from './ResultCard'

export type HistoryItem = {
  id: string
  videoName: string
  createdAt: string
  result?: ResultData
}

type HistorySidebarProps = {
  items: HistoryItem[]
  onSelect: (item: HistoryItem) => void
}

export function HistorySidebar({ items, onSelect }: HistorySidebarProps) {
  if (items.length === 0) return null

  return (
    <aside className="rounded-2xl border border-blue-800/50 bg-blue-950/40 shadow-xl shadow-blue-950/20 overflow-hidden">
      <div className="px-4 py-3 border-b border-blue-800/50">
        <h3 className="text-sm font-semibold text-blue-100">
          Previous results
        </h3>
      </div>
      <ul className="divide-y divide-blue-800/40 max-h-[320px] overflow-y-auto">
        {items.map((item) => (
          <li key={item.id}>
            <button
              type="button"
              onClick={() => onSelect(item)}
              className="w-full px-4 py-3 text-left text-sm hover:bg-blue-900/50 transition-colors"
            >
              <p className="font-medium text-blue-100 truncate">
                {item.videoName}
              </p>
              <p className="text-xs text-blue-200/60 mt-0.5">
                {new Date(item.createdAt).toLocaleString()}
              </p>
            </button>
          </li>
        ))}
      </ul>
    </aside>
  )
}
