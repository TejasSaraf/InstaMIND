function LogoIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden
    >
      <rect
        x="2"
        y="2"
        width="28"
        height="28"
        rx="7"
        stroke="currentColor"
        strokeWidth="2"
        fill="none"
      />
      <circle cx="16" cy="14" r="4" stroke="currentColor" strokeWidth="2" fill="none" />
      <path
        d="M16 6v2M16 22v2M10 10l1.5 1.5M20.5 20.5L22 22M22 10l-1.5 1.5M10.5 20.5L9 22"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  )
}

export function Navbar() {
  return (
    <nav className="sticky top-0 z-20 border-b border-neutral-800 bg-black/95 backdrop-blur-md shadow-lg shadow-black/20">
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        <div className="flex items-center h-14 sm:h-16">
          <div className="flex items-center gap-3 text-[var(--color-brand)]">
            <span className="w-10 h-10 rounded-xl bg-neutral-900 border border-neutral-700 flex items-center justify-center shrink-0 [color:var(--color-brand)]">
              <LogoIcon className="w-6 h-6" />
            </span>
            <span className="text-2xl sm:text-3xl font-bold tracking-tight text-[var(--color-brand)]">
              InstaMIND
            </span>
          </div>
        </div>
      </div>
    </nav>
  )
}
