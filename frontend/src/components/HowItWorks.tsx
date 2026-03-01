function StepBadge({ step }: { step: number }) {
  return (
    <span className="text-sm font-bold text-teal-400">{step}</span>
  )
}

const steps = [
  {
    step: 1,
    title: 'Upload',
    description: 'Drag & drop a video or click to choose a file. We support common video formats.',
  },
  {
    step: 2,
    title: 'Analyze',
    description: 'Hit Analyze to process your video and get insights.',
  },
  {
    step: 3,
    title: 'Get insights',
    description: 'View the summary and key insights. Copy results or revisit them from history.',
  },
]

export function HowItWorks() {
  return (
    <section className="rounded-2xl border border-neutral-800 bg-neutral-950 px-6 py-8 shadow-xl">
      <h2 className="text-lg font-semibold text-white mb-6 text-center">
        How it works
      </h2>
      <div className="grid sm:grid-cols-3 gap-6">
        {steps.map(({ step, title, description }) => (
          <div
            key={step}
            className="relative rounded-xl border border-neutral-800 bg-neutral-900/50 p-4"
          >
            <div className="w-8 h-8 rounded-lg bg-neutral-800 border border-neutral-600 text-teal-400 flex items-center justify-center mb-3">
              <StepBadge step={step} />
            </div>
            <h3 className="font-medium text-white mb-1">{title}</h3>
            <p className="text-sm text-neutral-400 leading-relaxed">
              {description}
            </p>
          </div>
        ))}
      </div>
    </section>
  )
}
