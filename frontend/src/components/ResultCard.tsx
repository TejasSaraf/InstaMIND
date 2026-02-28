import { useState } from 'react'

export type ResultData = {
  summary: string
  insights: string[]
  videoName?: string
}

type ResultCardProps = {
  data: ResultData
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <button
      type="button"
      onClick={handleCopy}
      className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium text-blue-200 hover:text-white bg-blue-900/60 hover:bg-blue-800/80 border border-blue-700/40 transition-colors"
    >
      {copied ? (
        <>
          <svg className="w-3.5 h-3.5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
          </svg>
          Copied
        </>
      ) : (
        <>
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
          Copy
        </>
      )}
    </button>
  )
}

export function ResultCard({ data }: ResultCardProps) {
  const fullText = [data.summary, ...data.insights].join('\n\n')

  return (
    <div className="rounded-2xl border border-blue-800/50 bg-blue-950/40 shadow-xl shadow-blue-950/20 overflow-hidden">
      <div className="px-5 py-4 border-b border-blue-800/50 flex items-center justify-between">
        <h3 className="font-semibold text-white">
          Analysis result
          {data.videoName && (
            <span className="text-blue-200/80 font-normal ml-2 truncate max-w-[200px] inline-block align-bottom">
              · {data.videoName}
            </span>
          )}
        </h3>
        <CopyButton text={fullText} />
      </div>

      <div className="p-5 space-y-5">
        <section>
          <h4 className="text-xs font-semibold uppercase tracking-wider text-blue-300/80 mb-2">
            Summary
          </h4>
          <p className="text-blue-100/90 leading-relaxed">
            {data.summary}
          </p>
        </section>

        <section>
          <h4 className="text-xs font-semibold uppercase tracking-wider text-blue-300/80 mb-2">
            Key insights
          </h4>
          <ul className="space-y-2">
            {data.insights.map((insight, i) => (
              <li key={i} className="flex gap-2 text-blue-100/90">
                <span className="text-blue-400 mt-0.5 shrink-0">•</span>
                <span className="leading-relaxed">{insight}</span>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  )
}
