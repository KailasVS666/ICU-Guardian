import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Heart, Droplets, Moon, AlertCircle, Wifi, WifiOff, Play, Square, Activity, FileText, ArrowLeft } from 'lucide-react'
import { useVisionControl, useMonitoringControl } from '../hooks/useSystemStatus'
import { Card, Button, Badge, AmbientBackground } from './ui'
import MetricCard from './MetricCard'
import AlertFeed from './AlertFeed'
import VitalsChart from './VitalsChart'
import PillarStatus from './PillarStatus'
import { generateClinicalReport } from '../utils/generateClinicalReport'

// Interface for tracking vitals readings over time
interface VitalsReading {
  timestamp: number
  hr: number
  spo2: number
  rass: number
  sleepScore: number
  riskProbability: number
}

interface DashboardProps {
  isConnected: boolean
  vitals: any
  alerts: any
  systemStatus: any
}

export default function Dashboard({ isConnected, vitals, alerts, systemStatus }: DashboardProps) {
  const navigate = useNavigate()
  const { startVision, stopVision } = useVisionControl()
  const { startMonitoring, stopMonitoring } = useMonitoringControl()
  const [isGeneratingPDF, setIsGeneratingPDF] = useState(false)

  // Track session start time and vitals history for averaging
  const sessionStartTime = useRef<Date>(new Date())
  const vitalsHistory = useRef<VitalsReading[]>([])

  const visionRunning = systemStatus?.vision_running || false
  const monitoringActive = systemStatus?.monitoring_active || false

  // Track vitals readings over time for session averaging
  useEffect(() => {
    if (vitals?.vitals) {
      const reading: VitalsReading = {
        timestamp: Date.now(),
        hr: vitals.vitals.HR_Avg || 72,
        spo2: vitals.vitals.SpO2_Min || 98,
        rass: vitals.vitals.RASS_Score || 0,
        sleepScore: vitals.vitals.Sleep_Score || 85,
        riskProbability: vitals.prediction?.probability || 0,
      }
      vitalsHistory.current.push(reading)
      
      // Keep last 1000 readings to prevent memory issues (approx 16+ hours at 1 reading/min)
      if (vitalsHistory.current.length > 1000) {
        vitalsHistory.current = vitalsHistory.current.slice(-1000)
      }
    }
  }, [vitals])

  // Calculate session averages from collected readings
  const calculateSessionAverages = () => {
    const readings = vitalsHistory.current
    if (readings.length === 0) {
      // Return current values if no history
      return {
        hr: vitals?.vitals?.HR_Avg || 72,
        spo2: vitals?.vitals?.SpO2_Min || 98,
        rass: vitals?.vitals?.RASS_Score || 0,
        sleepScore: vitals?.vitals?.Sleep_Score || 85,
        riskProbability: vitals?.prediction?.probability || 0,
        readingCount: 1,
        hrHistory: [vitals?.vitals?.HR_Avg || 72],
        spo2History: [vitals?.vitals?.SpO2_Min || 98],
      }
    }

    const sum = readings.reduce((acc, r) => ({
      hr: acc.hr + r.hr,
      spo2: acc.spo2 + r.spo2,
      rass: acc.rass + r.rass,
      sleepScore: acc.sleepScore + r.sleepScore,
      riskProbability: acc.riskProbability + r.riskProbability,
    }), { hr: 0, spo2: 0, rass: 0, sleepScore: 0, riskProbability: 0 })

    const count = readings.length

    return {
      hr: Math.round(sum.hr / count),
      spo2: Math.round(sum.spo2 / count),
      rass: Math.round(sum.rass / count),
      sleepScore: Math.round(sum.sleepScore / count),
      riskProbability: sum.riskProbability / count,
      readingCount: count,
      hrHistory: readings.map(r => r.hr),
      spo2History: readings.map(r => r.spo2),
    }
  }

  // PDF Clinical Report Generator - uses session averages
  const handleDownloadReport = async () => {
    setIsGeneratingPDF(true)
    try {
      // Calculate averages from session start to now
      const sessionData = calculateSessionAverages()
      const sessionDuration = Math.round((Date.now() - sessionStartTime.current.getTime()) / 1000)
      const sessionMinutes = Math.floor(sessionDuration / 60)
      const sessionHours = Math.floor(sessionMinutes / 60)
      
      const averageVitals = {
        hr: sessionData.hr,
        spo2: sessionData.spo2,
        rass: sessionData.rass,
        sleepScore: sessionData.sleepScore,
        riskProbability: sessionData.riskProbability,
      }

      // Calculate risk level from average probability
      const riskLevel = sessionData.riskProbability > 0.7 ? 'HIGH' : sessionData.riskProbability > 0.4 ? 'MODERATE' : 'LOW'

      // Get all alerts from session (matching AlertData interface)
      // Handle both array and object with alerts property
      const alertsArray = Array.isArray(alerts) ? alerts : (alerts?.alerts || alerts?.data || [])
      const sessionAlerts = (Array.isArray(alertsArray) ? alertsArray : []).map((alert: any) => ({
        timestamp: new Date(alert.timestamp || Date.now()).toISOString(),
        pillar: alert.pillar || 'System',
        message: alert.message || 'System alert',
        severity: alert.level || 'warning',
      }))

      // Generate AI observation based on session averages
      const sessionTimeStr = sessionHours > 0 
        ? `${sessionHours}h ${sessionMinutes % 60}m` 
        : `${sessionMinutes}m`
      
      const aiObservations = [
        `Session Summary: ${sessionData.readingCount} readings collected over ${sessionTimeStr}.`,
        `Average heart rate: ${averageVitals.hr} BPM ${averageVitals.hr > 80 ? '(elevated trend)' : '(stable)'}.`,
        `Average SpO2: ${averageVitals.spo2}% ${averageVitals.spo2 >= 95 ? '(optimal)' : '(requires attention)'}.`,
        `Average RASS Score: ${averageVitals.rass} - ${averageVitals.rass === 0 ? 'Alert and calm' : averageVitals.rass < 0 ? 'Sedated' : 'Agitated'}.`,
        `Vision system was ${visionRunning ? 'active during session' : 'on standby'}.`,
        `Overall risk assessment: ${riskLevel} (${(sessionData.riskProbability * 100).toFixed(1)}% avg probability).`,
        `Total alerts during session: ${sessionAlerts.length}.`,
      ]

      // Use actual history arrays from session or sample if too large
      let hrHistory = sessionData.hrHistory
      let spo2History = sessionData.spo2History
      
      // Sample down to 48 points if we have more (for PDF visualization)
      if (hrHistory.length > 48) {
        const step = Math.floor(hrHistory.length / 48)
        hrHistory = hrHistory.filter((_, i) => i % step === 0).slice(0, 48)
        spo2History = spo2History.filter((_, i) => i % step === 0).slice(0, 48)
      }

      // Generate and download the PDF
      generateClinicalReport({
        vitals: averageVitals,
        alerts: sessionAlerts.slice(0, 10), // Latest 10 alerts
        aiMessage: aiObservations.join('\n'),
        hrHistory,
        spo2History,
        uptime: systemStatus?.uptime || '00:00:00',
      })
    } catch (error) {
      console.error('Failed to generate PDF report:', error)
    } finally {
      setIsGeneratingPDF(false)
    }
  }

  return (
    <>
      {/* Ambient Background - Floating gradient blobs */}
      <AmbientBackground />
      
      <div className="relative z-10 min-h-screen p-4 md:p-6 lg:p-8">
        {/* Hero Header */}
        <Card className="mb-6 animate-fade-in" padding="lg">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
            {/* Left - Title & Description */}
            <div className="flex-1">
              {/* Back Button & Label */}
              <div className="flex items-center gap-3 mb-3">
                <button
                  onClick={() => navigate('/')}
                  className="group flex items-center justify-center w-8 h-8 rounded-lg bg-white/[0.03] border border-white/[0.06] hover:bg-white/[0.08] hover:border-accent/30 transition-all duration-200"
                  aria-label="Back to Home"
                >
                  <ArrowLeft className="w-4 h-4 text-foreground-muted group-hover:text-accent transition-colors" />
                </button>
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-accent" />
                  <span className="text-xs font-mono uppercase tracking-widest text-foreground-muted">
                    ICU Guardian · Command Center
                  </span>
                </div>
              </div>
              
              {/* Hero Title */}
              <h1 className="text-3xl md:text-4xl lg:text-5xl font-semibold tracking-tight mb-3">
                <span className="text-gradient">Live Ops &</span>{' '}
                <span className="text-gradient-accent">Clinical AI</span>
              </h1>
              
              {/* Description */}
              <p className="text-foreground-muted text-base md:text-lg max-w-2xl leading-relaxed">
                Continuous bedside vision, real-time vitals, and AI-driven decision support 
                aligned into a single control surface built for rapid response.
              </p>
              
              {/* Status Chips */}
              <div className="flex flex-wrap gap-3 mt-5">
                <div className="flex items-center gap-2 px-4 py-2 bg-white/[0.03] rounded-xl border border-white/[0.06]">
                  <span className="text-foreground-muted text-sm">Uptime</span>
                  <span className="text-accent font-semibold text-sm font-mono">
                    {systemStatus?.uptime || '00:00:00'}
                  </span>
                </div>
                <div className="flex items-center gap-2 px-4 py-2 bg-white/[0.03] rounded-xl border border-white/[0.06]">
                  <span className="text-foreground-muted text-sm">Clients</span>
                  <span className="text-accent font-semibold text-sm font-mono">
                    {systemStatus?.connected_clients || 0}
                  </span>
                </div>
              </div>
            </div>

            {/* Right - Connection Status & Controls */}
            <div className="flex flex-col items-start lg:items-end gap-4">
              {/* Connection Badge */}
              {isConnected ? (
                <Badge variant="success" pulse>
                  <Wifi className="w-3 h-3" />
                  Connected
                </Badge>
              ) : (
                <Badge variant="error" pulse>
                  <WifiOff className="w-3 h-3" />
                  Disconnected
                </Badge>
              )}

              {/* Control Buttons */}
              <div className="flex flex-wrap gap-3">
                {visionRunning ? (
                  <Button
                    variant="danger"
                    size="md"
                    leftIcon={<Square className="w-4 h-4" />}
                    onClick={() => stopVision.mutate()}
                  >
                    Stop Vision
                  </Button>
                ) : (
                  <Button
                    variant="primary"
                    size="md"
                    leftIcon={<Play className="w-4 h-4" />}
                    onClick={() => startVision.mutate()}
                  >
                    Start Vision
                  </Button>
                )}

                {monitoringActive ? (
                  <Button
                    variant="secondary"
                    size="md"
                    onClick={() => stopMonitoring.mutate()}
                  >
                    Stop Monitoring
                  </Button>
                ) : (
                  <Button
                    variant="secondary"
                    size="md"
                    onClick={() => startMonitoring.mutate()}
                  >
                    Start Monitoring
                  </Button>
                )}

                {/* PDF Clinical Report Download */}
                <Button
                  variant="secondary"
                  size="md"
                  leftIcon={isGeneratingPDF ? (
                    <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <FileText className="w-4 h-4" />
                  )}
                  onClick={handleDownloadReport}
                  disabled={isGeneratingPDF}
                >
                  {isGeneratingPDF ? 'Generating...' : 'Clinical Report'}
                </Button>
              </div>
            </div>
          </div>
          
          {/* Divider line */}
          <div className="monitor-line mt-6" />
        </Card>

        {/* Monitoring Status Indicator */}
        {!monitoringActive && (
          <div className="flex items-center justify-center gap-3 mb-4 py-3 px-5 rounded-xl bg-yellow-500/10 border border-yellow-500/20">
            <div className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-yellow-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-3 w-3 bg-yellow-500" />
            </div>
            <span className="text-yellow-400 text-sm font-medium">
              Monitoring Paused — Readings are not being tracked
            </span>
          </div>
        )}

        {/* Metrics Row - Staggered Animation */}
        <div className={`grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6 stagger-children transition-all duration-500 ${!monitoringActive ? 'blur-md pointer-events-none select-none' : ''}`}>
          <MetricCard
            icon={<Heart className="w-5 h-5" />}
            label="Heart Rate"
            value={vitals?.vitals?.HR_Avg?.toFixed(0) || '--'}
            unit="BPM"
            status={
              vitals?.vitals?.HR_Avg > 100 || vitals?.vitals?.HR_Avg < 60
                ? 'warning'
                : 'normal'
            }
            accentColor="text-red-400"
          />
          <MetricCard
            icon={<Droplets className="w-5 h-5" />}
            label="SpO2"
            value={vitals?.vitals?.SpO2_Min?.toFixed(0) || '--'}
            unit="%"
            status={
              vitals?.vitals?.SpO2_Min < 92
                ? 'critical'
                : vitals?.vitals?.SpO2_Min < 95
                ? 'warning'
                : 'normal'
            }
            accentColor="text-blue-400"
          />
          <MetricCard
            icon={<Moon className="w-5 h-5" />}
            label="Sleep Score"
            value={vitals?.vitals?.Sleep_Score?.toFixed(1) || '--'}
            unit="/5"
            status="normal"
            accentColor="text-violet-400"
          />
          <MetricCard
            icon={<AlertCircle className="w-5 h-5" />}
            label="Risk Level"
            value={vitals ? `${(vitals.prediction.probability * 100).toFixed(0)}%` : '--'}
            unit=""
            status={
              vitals?.prediction.probability > 0.6
                ? 'critical'
                : vitals?.prediction.probability > 0.4
                ? 'warning'
                : 'normal'
            }
            accentColor={
              vitals?.prediction.probability > 0.6
                ? 'text-red-400'
                : vitals?.prediction.probability > 0.4
                ? 'text-yellow-400'
                : 'text-green-400'
            }
          />
        </div>

        {/* Main Content Grid - Bento Layout */}
        <div className={`grid grid-cols-1 lg:grid-cols-6 gap-6 transition-all duration-500 ${!monitoringActive ? 'blur-md pointer-events-none select-none' : ''}`}>
          {/* Charts - Spans 4 columns */}
          <div className="lg:col-span-4 space-y-6">
            <VitalsChart vitals={vitals} />
            <PillarStatus alerts={alerts} />
          </div>

          {/* Alert Feed - Spans 2 columns */}
          <div className="lg:col-span-2">
            <AlertFeed alerts={alerts} />
          </div>
        </div>
      </div>
    </>
  )
}
