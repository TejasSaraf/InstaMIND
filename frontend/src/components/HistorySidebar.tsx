import type { ResultData } from './ResultCard'

function IconHistory({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  )
}

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
    <aside className="rounded-2xl border border-neutral-800 bg-neutral-950 shadow-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-neutral-800 flex items-center gap-2">
        <IconHistory className="w-4 h-4 text-neutral-500 shrink-0" />
        <h3 className="text-sm font-semibold text-neutral-200">
          Previous results
        </h3>
      </div>
      <ul className="divide-y divide-neutral-800 max-h-[320px] overflow-y-auto">
        {items.map((item) => (
          <li key={item.id}>
            <button
              type="button"
              onClick={() => onSelect(item)}
              className="w-full px-4 py-3 text-left text-sm hover:bg-neutral-800/50 transition-colors"
            >
              <p className="font-medium text-neutral-200 truncate">
                {item.videoName}
              </p>
              <p className="text-xs text-neutral-500 mt-0.5">
                {new Date(item.createdAt).toLocaleString()}
              </p>
            </button>
          </li>
        ))}
      </ul>
    </aside>
  )
}
