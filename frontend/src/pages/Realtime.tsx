export function Realtime() {
  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8">
      <div className="rounded-2xl border border-blue-800/50 bg-blue-950/40 shadow-xl shadow-blue-950/20 p-8 sm:p-10 text-center">
        <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-blue-900/80 border border-blue-600/50 text-blue-300 mb-5">
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
            <circle cx="12" cy="12" r="10" />
            <path d="M12 6v6l4 2" />
          </svg>
        </div>
        <h2 className="text-xl font-semibold text-white mb-2">
          Real time
        </h2>
        <p className="text-blue-200/80 text-sm max-w-sm mx-auto leading-relaxed">
          Live streaming and real-time features will appear here. Connect feeds and see updates as they happen.
        </p>
      </div>
    </div>
  )
}
