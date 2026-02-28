import type { SystemStatus, Incident, Reasoning, AnalyticsSummary } from '../types'

export const MOCK_STATUS: SystemStatus = 'ALERT'

export const MOCK_INCIDENTS: Incident[] = [
  { id: '1', timestamp: '14:32:05', eventType: 'fall', eventLabel: 'Fall Detected', status: 'ALERT' },
  { id: '2', timestamp: '14:28:12', eventType: 'choking', eventLabel: 'Choking Risk', status: 'MONITOR' },
  { id: '3', timestamp: '14:15:33', eventType: 'suspicious', eventLabel: 'Suspicious Activity', status: 'Monitoring' },
  { id: '4', timestamp: '14:02:41', eventType: 'fall', eventLabel: 'Fall Detected', status: 'ALERT' },
  { id: '5', timestamp: '13:58:22', eventType: 'shoplifting', eventLabel: 'Shoplifting Alert', status: 'ALERT' },
  { id: '6', timestamp: '13:45:10', eventType: 'suspicious', eventLabel: 'Loitering', status: 'Normal' },
]

export const MOCK_REASONING: Record<string, Reasoning> = {
  '1': {
    eventName: 'Fall Detected',
    eventType: 'fall',
    confidence: 0.94,
    reasons: [
      'Sudden vertical displacement detected in pose keypoints',
      'Prolonged immobility (>3s) after impact',
      'Subject position indicates person on floor',
    ],
    decision: 'ALERT',
  },
  '2': {
    eventName: 'Choking Risk',
    eventType: 'choking',
    confidence: 0.87,
    reasons: [
      'Hand-to-throat gesture detected',
      'Distressed facial expression classification',
      'Duration exceeds safety threshold',
    ],
    decision: 'MONITOR',
  },
  '3': {
    eventName: 'Suspicious Activity',
    eventType: 'suspicious',
    confidence: 0.72,
    reasons: [
      'Unusual loitering pattern near restricted area',
      'Multiple direction changes without purpose',
      'No immediate threat indicators',
    ],
    decision: 'MONITOR',
  },
  '4': {
    eventName: 'Fall Detected',
    eventType: 'fall',
    confidence: 0.91,
    reasons: [
      'Rapid descent from standing position',
      'Impact signature in motion analysis',
      'Subject not responsive for 3+ seconds',
    ],
    decision: 'ALERT',
  },
  '5': {
    eventName: 'Shoplifting Alert',
    eventType: 'shoplifting',
    confidence: 0.89,
    reasons: [
      'Item concealed in bag without scan',
      'Behavior consistent with theft pattern',
      'Rapid exit toward door',
    ],
    decision: 'ALERT',
  },
  '6': {
    eventName: 'Loitering',
    eventType: 'suspicious',
    confidence: 0.65,
    reasons: [
      'Extended stay in single zone',
      'No purchase intent signals',
      'Classified as low priority',
    ],
    decision: 'MONITOR',
  },
}

export const MOCK_ANALYTICS: AnalyticsSummary = {
  totalIncidents: 127,
  alertsToday: 8,
  eventDistribution: [
    { label: 'Fall', count: 42, color: 'bg-red-500' },
    { label: 'Suspicious', count: 38, color: 'bg-amber-500' },
    { label: 'Shoplifting', count: 28, color: 'bg-rose-500' },
    { label: 'Choking', count: 19, color: 'bg-orange-500' },
  ],
}
