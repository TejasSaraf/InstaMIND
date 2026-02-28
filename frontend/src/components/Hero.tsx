type HeroProps = {
  onGetStarted: () => void
}

export function Hero({ onGetStarted }: HeroProps) {
  return (
    <section className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-blue-950/90 via-blue-950/70 to-black border border-blue-800/50 px-6 py-12 sm:px-10 sm:py-16 text-center shadow-xl shadow-blue-950/30">
      <div className="relative z-10">
        <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-white mb-3">
          Understand your videos in seconds
        </h1>
        <p className="text-blue-200/90 text-lg max-w-xl mx-auto mb-8">
          Upload any video and get instant summaries and insights. Perfect for content creators and teams.
        </p>
        <button
          type="button"
          onClick={onGetStarted}
          className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-medium shadow-lg shadow-blue-500/30 hover:shadow-blue-500/40 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-offset-2 focus:ring-offset-black"
        >
          Get Started
        </button>
      </div>
    </section>
  )
}
