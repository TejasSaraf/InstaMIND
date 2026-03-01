function IconSparkles({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
    </svg>
  )
}

type HeroProps = {
  onGetStarted: () => void
}

export function Hero({ onGetStarted }: HeroProps) {
  return (
    <section className="relative overflow-hidden rounded-2xl bg-neutral-950 border border-neutral-800 px-6 py-12 sm:px-10 sm:py-16 text-center shadow-xl">
      <div className="relative z-10">
        <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-neutral-800 border border-neutral-700 text-teal-400 mb-4">
          <IconSparkles className="w-6 h-6" />
        </div>
        <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-white mb-3">
          Understand your videos in seconds
        </h1>
        <p className="text-neutral-400 text-lg max-w-xl mx-auto mb-8">
          Upload any video and get instant summaries and insights. Perfect for content creators and teams.
        </p>
        <button
          type="button"
          onClick={onGetStarted}
          className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-teal-600 hover:bg-teal-500 text-white font-medium border border-teal-500/50 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-teal-400 focus:ring-offset-2 focus:ring-offset-black"
        >
          Get Started
        </button>
      </div>
    </section>
  )
}
