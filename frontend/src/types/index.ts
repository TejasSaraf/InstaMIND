export type SystemStatus = 'Normal' | 'Monitoring' | 'ALERT'

export type Incident = {
  id: string
  timestamp: string
  eventType: 'fall' | 'shoplifting' | 'suspicious' | 'choking'
  eventLabel: string
  status: SystemStatus
}

export type Reasoning = {
  eventName: string
  eventType: string
  confidence: number
  reasons: string[]
  decision: 'ALERT' | 'MONITOR'
}

export type AnalyticsSummary = {
  totalIncidents: number
  alertsToday: number
  eventDistribution: { label: string; count: number; color: string }[]
}
