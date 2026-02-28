import { useTheme } from '../contexts/ThemeContext'

type NavItem = 'upload' | 'realtime'

type NavbarProps = {
  active: NavItem
  onNavigate: (item: NavItem) => void
}

export function Navbar({ active, onNavigate }: NavbarProps) {
  const { theme, toggleTheme } = useTheme()

  const linkBase =
    'px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200'
  const linkInactive =
    'text-blue-200/80 hover:text-white hover:bg-blue-950/80'
  const linkActive =
    'text-blue-300 bg-blue-950/90 border border-blue-700/60'

  return (
    <nav className="sticky top-0 z-20 border-b border-blue-900/50 bg-black/95 backdrop-blur-md shadow-lg shadow-blue-950/20">
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        <div className="flex items-center justify-between h-14 sm:h-16">
          <button
            type="button"
            onClick={() => onNavigate('upload')}
            className="flex items-center gap-2 text-lg font-semibold tracking-tight text-white hover:text-blue-300 transition-colors"
          >
            <span className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white text-sm font-bold shadow-lg shadow-blue-500/30">
              IM
            </span>
            InstaMIND
          </button>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => onNavigate('upload')}
              className={`${linkBase} ${active === 'upload' ? linkActive : linkInactive}`}
            >
              Upload Video
            </button>
            <button
              type="button"
              onClick={() => onNavigate('realtime')}
              className={`${linkBase} ${active === 'realtime' ? linkActive : linkInactive}`}
            >
              Real time
            </button>

            <div className="w-px h-6 bg-blue-800/60 mx-1" />

            <button
              type="button"
              onClick={toggleTheme}
              className="p-2 rounded-lg text-blue-200/80 hover:text-white hover:bg-blue-950/80 transition-colors"
              title={theme === 'dark' ? 'Switch to light' : 'Switch to dark'}
              aria-label="Toggle theme"
            >
              {theme === 'dark' ? (
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                </svg>
              )}
            </button>

            <div className="hidden sm:flex items-center gap-2 pl-2">
              <div className="w-8 h-8 rounded-full bg-blue-900/80 border border-blue-700/50 flex items-center justify-center text-blue-200 text-sm font-medium">
                U
              </div>
              <span className="text-sm text-blue-200/70 hidden md:inline">
                Profile
              </span>
            </div>
          </div>
        </div>
      </div>
    </nav>
  )
}
