import { useState, useEffect, useRef, MouseEvent } from 'react'
import { format } from 'date-fns'
import { AlertTriangle, CheckCircle, Info, XCircle, Bell } from 'lucide-react'

interface AlertFeedProps {
  alerts: any
}

// Gentle alert sound using Web Audio API
function playAlertSound(severity: 'critical' | 'warning' | 'info' = 'warning') {
  try {
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)()
    
    // Create oscillator for gentle tone
    const oscillator = audioContext.createOscillator()
    const gainNode = audioContext.createGain()
    
    // Different frequencies for different severities
    const frequencies = {
      critical: [440, 523], // A4 to C5 - more urgent
      warning: [392, 440],  // G4 to A4 - moderate
      info: [349, 392]      // F4 to G4 - gentle
    }
    
    const [startFreq, endFreq] = frequencies[severity]
    
    oscillator.type = 'sine'
    oscillator.frequency.setValueAtTime(startFreq, audioContext.currentTime)
    oscillator.frequency.exponentialRampToValueAtTime(endFreq, audioContext.currentTime + 0.15)
    
    // Gentle volume envelope
    gainNode.gain.setValueAtTime(0, audioContext.currentTime)
    gainNode.gain.linearRampToValueAtTime(0.15, audioContext.currentTime + 0.05) // Soft attack
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.4) // Gentle decay
    
    oscillator.connect(gainNode)
    gainNode.connect(audioContext.destination)
    
    oscillator.start(audioContext.currentTime)
    oscillator.stop(audioContext.currentTime + 0.4)
    
    // Second tone for critical alerts
    if (severity === 'critical') {
      setTimeout(() => {
        const osc2 = audioContext.createOscillator()
        const gain2 = audioContext.createGain()
        osc2.type = 'sine'
        osc2.frequency.setValueAtTime(523, audioContext.currentTime)
        osc2.frequency.exponentialRampToValueAtTime(659, audioContext.currentTime + 0.15)
        gain2.gain.setValueAtTime(0, audioContext.currentTime)
        gain2.gain.linearRampToValueAtTime(0.12, audioContext.currentTime + 0.05)
        gain2.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3)
        osc2.connect(gain2)
        gain2.connect(audioContext.destination)
        osc2.start(audioContext.currentTime)
        osc2.stop(audioContext.currentTime + 0.3)
      }, 200)
    }
  } catch (e) {
    // Audio not supported or blocked
    console.log('Alert sound not available')
  }
}

export default function AlertFeed({ alerts }: AlertFeedProps) {
  const [alertHistory, setAlertHistory] = useState<any[]>([])
  const cardRef = useRef<HTMLDivElement>(null)
  const lastAlertTimeRef = useRef<string | null>(null)

  const handleMouseMove = (e: MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return
    const rect = cardRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    cardRef.current.style.setProperty('--mouse-x', `${x}px`)
    cardRef.current.style.setProperty('--mouse-y', `${y}px`)
  }

  useEffect(() => {
    if (alerts?.pillars) {
      Object.entries(alerts.pillars).forEach(([name, data]: [string, any]) => {
        if (data.active && data.last_alert) {
          const exists = alertHistory.some(
            (a) => a.pillar === name && a.timestamp === data.last_alert
          )

          if (!exists) {
            const severity = name === 'self_harm' || name === 'fall' ? 'critical' : 'warning'
            
            // Play alert sound for new alerts
            if (lastAlertTimeRef.current !== data.last_alert) {
              playAlertSound(severity)
              lastAlertTimeRef.current = data.last_alert
            }
            
            setAlertHistory((prev) => [
              {
                pillar: name,
                message: `Alert detected`,
                timestamp: data.last_alert || new Date().toISOString(),
                severity,
              },
              ...prev.slice(0, 49),
            ])
          }
        }
      })
    }
  }, [alerts])

  const severityConfig = {
    critical: { 
      bg: 'bg-red-500/[0.08]', 
      border: 'border-l-red-500', 
      text: 'text-red-400',
      icon: <XCircle className="w-5 h-5" />,
      glow: 'hover:shadow-[0_0_30px_rgba(239,68,68,0.1)]'
    },
    warning: { 
      bg: 'bg-yellow-500/[0.08]', 
      border: 'border-l-yellow-500', 
      text: 'text-yellow-400',
      icon: <AlertTriangle className="w-5 h-5" />,
      glow: 'hover:shadow-[0_0_30px_rgba(234,179,8,0.1)]'
    },
    info: { 
      bg: 'bg-accent/[0.08]', 
      border: 'border-l-accent', 
      text: 'text-accent-bright',
      icon: <Info className="w-5 h-5" />,
      glow: 'hover:shadow-[0_0_30px_rgba(94,106,210,0.1)]'
    },
    success: { 
      bg: 'bg-green-500/[0.08]', 
      border: 'border-l-green-500', 
      text: 'text-green-400',
      icon: <CheckCircle className="w-5 h-5" />,
      glow: 'hover:shadow-[0_0_30px_rgba(34,197,94,0.1)]'
    },
  }

  return (
    <div 
      ref={cardRef}
      onMouseMove={handleMouseMove}
      className="glass-card spotlight-card p-6 h-[600px] flex flex-col"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-accent/10 border border-accent/20">
            <Bell className="w-5 h-5 text-accent" />
          </div>
          <div>
            <h3 className="text-lg font-semibold tracking-tight text-foreground">
              Alert Feed
            </h3>
            <p className="text-xs text-foreground-muted">
              {alertHistory.length} alerts recorded
            </p>
          </div>
        </div>
        
        {/* Live indicator */}
        <div className="flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-accent" />
          </span>
          <span className="text-xs font-mono text-foreground-muted uppercase tracking-widest">
            Live
          </span>
        </div>
      </div>
      
      {/* Alert List */}
      <div className="flex-1 overflow-y-auto space-y-3 pr-1 -mr-1">
        {alertHistory.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center py-8">
            <div className="p-4 rounded-2xl bg-green-500/10 border border-green-500/20 mb-4">
              <CheckCircle className="w-8 h-8 text-green-400" />
            </div>
            <p className="text-foreground font-medium mb-1">All Clear</p>
            <p className="text-foreground-muted text-sm">No alerts recorded</p>
            <p className="text-foreground-subtle text-xs mt-2">System monitoring active</p>
          </div>
        ) : (
          alertHistory.map((alert, index) => {
            const config = severityConfig[alert.severity as keyof typeof severityConfig] || severityConfig.info
            
            return (
              <div
                key={index}
                className={`
                  p-4 rounded-xl border-l-4 
                  ${config.bg} ${config.border}
                  ${config.glow}
                  transition-all duration-200 ease-expo-out
                  hover:translate-x-1
                  animate-slide-up
                `}
                style={{ 
                  animationDelay: `${index * 50}ms`,
                  animationFillMode: 'both' 
                }}
              >
                <div className="flex items-start gap-3">
                  <div className={config.text}>
                    {config.icon}
                  </div>
                  <div className="flex-1 min-w-0">
                    <h4 className={`font-semibold text-sm ${config.text} uppercase tracking-wide`}>
                      {alert.pillar.replace('_', ' ')}
                    </h4>
                    <p className="text-foreground text-sm mt-1 truncate">
                      {alert.message}
                    </p>
                    <p className="text-foreground-muted text-xs mt-2 font-mono">
                      {format(new Date(alert.timestamp), 'HH:mm:ss')}
                    </p>
                  </div>
                </div>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
