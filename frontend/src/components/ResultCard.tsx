import { useState } from 'react'

export type ResultData = {
  summary: string
  insights: string[]
  videoName?: string
}

type ResultCardProps = {
  data: ResultData
}

function IconCopy({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
    </svg>
  )
}

function IconCheck({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
    </svg>
  )
}

function IconDocument({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
    </svg>
  )
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
      className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium text-neutral-300 hover:text-white bg-neutral-800 hover:bg-neutral-700 border border-neutral-600 transition-colors"
    >
      {copied ? (
        <>
          <IconCheck className="w-3.5 h-3.5 text-teal-400" />
          Copied
        </>
      ) : (
        <>
          <IconCopy className="w-3.5 h-3.5" />
          Copy
        </>
      )}
    </button>
  )
}

export function ResultCard({ data }: ResultCardProps) {
  const fullText = [data.summary, ...data.insights].join('\n\n')

  return (
    <div className="rounded-2xl border border-neutral-800 bg-neutral-950 shadow-xl overflow-hidden">
      <div className="px-5 py-4 border-b border-neutral-800 flex items-center justify-between">
        <h3 className="font-semibold text-white flex items-center gap-2">
          <IconDocument className="w-4 h-4 text-neutral-500 shrink-0" />
          Analysis result
          {data.videoName && (
            <span className="text-neutral-400 font-normal ml-2 truncate max-w-[200px] inline-block align-bottom">
              · {data.videoName}
            </span>
          )}
        </h3>
        <CopyButton text={fullText} />
      </div>

      <div className="p-5 space-y-5">
        <section>
          <h4 className="text-xs font-semibold uppercase tracking-wider text-neutral-500 mb-2">
            Summary
          </h4>
          <p className="text-neutral-300 leading-relaxed">
            {data.summary}
          </p>
        </section>

        <section>
          <h4 className="text-xs font-semibold uppercase tracking-wider text-neutral-500 mb-2">
            Key insights
          </h4>
          <ul className="space-y-2">
            {data.insights.map((insight, i) => (
              <li key={i} className="flex gap-2 text-neutral-300">
                <span className="text-neutral-500 mt-0.5 shrink-0">•</span>
                <span className="leading-relaxed">{insight}</span>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  )
}
