export function Realtime() {
  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8">
      <div className="rounded-2xl border border-neutral-800 bg-neutral-950 shadow-xl p-8 sm:p-10 text-center">
        <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-neutral-800 border border-neutral-600 text-teal-400 mb-5">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="w-7 h-7"
          >
            <path d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
        </div>
        <h2 className="text-xl font-semibold text-white mb-2">
          Real time
        </h2>
        <p className="text-neutral-400 text-sm max-w-sm mx-auto leading-relaxed">
          Live streaming and real-time features will appear here. Connect feeds and see updates as they happen.
        </p>
      </div>
    </div>
  )
}
