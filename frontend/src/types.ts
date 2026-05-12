export type SystemStatus = 'Normal' | 'Monitoring' | 'ALERT'

export type Incident = {
  id: string
  timestamp: string
  eventLabel: string
  status: SystemStatus
}

export type Reasoning = {
  eventName: string
  confidence: number
  reasons: string[]
  decision: string
}
