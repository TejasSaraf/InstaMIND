import { useState } from 'react'
import { ThemeProvider } from './contexts/ThemeContext'
import { Navbar } from './components/Navbar'
import { UploadVideo } from './pages/UploadVideo'
import { Realtime } from './pages/Realtime'

type NavItem = 'upload' | 'realtime'

function AppContent() {
  const [active, setActive] = useState<NavItem>('upload')

  return (
    <div className="min-h-screen bg-black text-white font-sans antialiased">
      <Navbar active={active} onNavigate={setActive} />
      <main className="py-6">
        {active === 'upload' && <UploadVideo />}
        {active === 'realtime' && <Realtime />}
      </main>
    </div>
  )
}

export default function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  )
}
