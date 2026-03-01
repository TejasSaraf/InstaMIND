import { Navbar } from './components/Navbar'
import { UploadVideo } from './pages/UploadVideo'

export default function App() {
  return (
    <div className="min-h-screen bg-black text-white font-sans antialiased">
      <Navbar />
      <main className="py-6">
        <UploadVideo />
      </main>
    </div>
  )
}
