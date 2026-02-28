import { MOCK_ANALYTICS } from '../data/mockData'

export function Analytics() {
  const { totalIncidents, alertsToday, eventDistribution } = MOCK_ANALYTICS

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <div className="rounded-xl border border-gray-800 bg-gray-800/50 p-5 shadow-lg">
          <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">
            Total incidents
          </p>
          <p className="text-2xl font-bold text-white">{totalIncidents}</p>
        </div>
        <div className="rounded-xl border border-gray-800 bg-gray-800/50 p-5 shadow-lg">
          <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">
            Alerts today
          </p>
          <p className="text-2xl font-bold text-red-400">{alertsToday}</p>
        </div>
        <div className="rounded-xl border border-gray-800 bg-gray-800/50 p-5 shadow-lg sm:col-span-2 lg:col-span-1">
          <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">
            Status
          </p>
          <p className="text-lg font-semibold text-white">Last 24h</p>
        </div>
      </div>

      <div className="rounded-xl border border-gray-800 bg-gray-800/50 p-6 shadow-xl">
        <h2 className="text-sm font-semibold text-white mb-4">Event distribution</h2>
        <div className="space-y-4">
          {eventDistribution.map((item) => (
            <div key={item.label} className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-300">{item.label}</span>
                <span className="text-gray-400 font-medium">{item.count}</span>
              </div>
              <div className="h-3 rounded-full bg-gray-700 overflow-hidden">
                <div
                  className={`h-full rounded-full ${item.color} transition-all duration-700`}
                  style={{ width: `${(item.count / totalIncidents) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="rounded-xl border border-gray-800 bg-gray-800/50 p-6 shadow-xl">
        <h2 className="text-sm font-semibold text-white mb-4">Distribution (simple pie style)</h2>
        <div className="flex flex-wrap gap-2 items-center justify-center py-4">
          {eventDistribution.map((item) => {
            const pct = (item.count / totalIncidents) * 100
            return (
              <div
                key={item.label}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-700/50"
              >
                <div
                  className={`w-3 h-3 rounded-full ${item.color}`}
                  style={{ minWidth: 12, minHeight: 12 }}
                />
                <span className="text-sm text-gray-300">{item.label}</span>
                <span className="text-sm text-gray-500 tabular-nums">{pct.toFixed(0)}%</span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
