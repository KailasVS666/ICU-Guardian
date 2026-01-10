import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Dashboard from './components/Dashboard'
import LandingPage from './pages/LandingPage'
import { useWebSocket } from './hooks/useWebSocket'
import { useSystemStatus } from './hooks/useSystemStatus'

function DashboardRoute() {
  const { isConnected, vitals, alerts } = useWebSocket()
  const { systemStatus } = useSystemStatus()

  return (
    <div className="min-h-screen bg-background-base text-foreground antialiased">
      <Dashboard 
        isConnected={isConnected}
        vitals={vitals}
        alerts={alerts}
        systemStatus={systemStatus}
      />
    </div>
  )
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/dashboard" element={<DashboardRoute />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
