import { useState } from 'react'
import { LiveFeed } from '../components/LiveFeed'
import { ReasonPanel } from '../components/ReasonPanel'
import { Timeline } from '../components/Timeline'
import { MOCK_INCIDENTS, MOCK_REASONING } from '../data/mockData'
import type { SystemStatus } from '../types'

type DashboardProps = {
  status: SystemStatus
}

export function Dashboard({ status }: DashboardProps) {
  const [selectedId, setSelectedId] = useState<string>(MOCK_INCIDENTS[0]?.id ?? '')
  const selectedReasoning = selectedId ? MOCK_REASONING[selectedId] ?? null : null

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-[7fr_3fr] gap-6">
        <div className="min-w-0">
          <LiveFeed status={status} />
        </div>
        <div className="min-w-0">
          <ReasonPanel data={selectedReasoning} />
        </div>
      </div>

      <Timeline
        incidents={MOCK_INCIDENTS}
        selectedId={selectedId}
        onSelect={setSelectedId}
      />
    </div>
  )
}
