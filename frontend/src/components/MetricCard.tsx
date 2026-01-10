import { useRef, MouseEvent, ReactNode } from 'react'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

type MetricStatus = 'normal' | 'warning' | 'critical'

interface MetricCardProps {
  icon: ReactNode
  label: string
  value: string | number
  unit: string
  status?: MetricStatus
  accentColor?: string
  trend?: 'up' | 'down' | 'stable'
  trendValue?: string
}

export default function MetricCard({ 
  icon, 
  label, 
  value, 
  unit, 
  status = 'normal',
  accentColor = 'text-accent',
  trend,
  trendValue
}: MetricCardProps) {
  const cardRef = useRef<HTMLDivElement>(null)

  const handleMouseMove = (e: MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return
    
    const rect = cardRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    
    cardRef.current.style.setProperty('--mouse-x', `${x}px`)
    cardRef.current.style.setProperty('--mouse-y', `${y}px`)
  }

  // Status-based glow colors
  const statusStyles = {
    normal: {
      glow: 'rgba(94, 106, 210, 0.15)',
      ring: 'border-white/[0.06]',
      indicator: 'bg-green-500'
    },
    warning: {
      glow: 'rgba(234, 179, 8, 0.15)',
      ring: 'border-yellow-500/30',
      indicator: 'bg-yellow-500'
    },
    critical: {
      glow: 'rgba(239, 68, 68, 0.2)',
      ring: 'border-red-500/30',
      indicator: 'bg-red-500'
    }
  }

  const currentStatus = statusStyles[status]

  return (
    <div
      ref={cardRef}
      onMouseMove={handleMouseMove}
      className={`
        glass-card spotlight-card p-5 relative overflow-hidden
        transition-all duration-300 ease-expo-out
        hover:transform hover:-translate-y-1
        ${currentStatus.ring}
      `}
      style={{
        '--spotlight-color': currentStatus.glow
      } as React.CSSProperties}
    >
      {/* Spotlight gradient overlay */}
      <div 
        className="absolute inset-0 opacity-0 hover:opacity-100 transition-opacity duration-300 pointer-events-none"
        style={{
          background: `radial-gradient(300px circle at var(--mouse-x, 50%) var(--mouse-y, 50%), ${currentStatus.glow}, transparent 70%)`
        }}
      />
      
      {/* Status indicator dot */}
      <div className={`absolute top-3 right-3 w-2 h-2 rounded-full ${currentStatus.indicator} ${status !== 'normal' ? 'animate-pulse' : ''}`} />
      
      {/* Header */}
      <div className="flex items-center justify-between mb-4 relative z-10">
        <span className="text-xs font-mono uppercase tracking-widest text-foreground-muted">
          {label}
        </span>
        <div className={`${accentColor} p-2 rounded-xl bg-white/[0.03] border border-white/[0.06]`}>
          {icon}
        </div>
      </div>
      
      {/* Value */}
      <div className="flex items-baseline gap-2 relative z-10">
        <span className={`text-4xl font-semibold tracking-tight ${accentColor}`}>
          {value}
        </span>
        <span className="text-foreground-muted text-lg font-medium">
          {unit}
        </span>
      </div>
      
      {/* Trend indicator (optional) */}
      {trend && (
        <div className="flex items-center gap-1.5 mt-3 relative z-10">
          {trend === 'up' && <TrendingUp className="w-4 h-4 text-green-400" />}
          {trend === 'down' && <TrendingDown className="w-4 h-4 text-red-400" />}
          {trend === 'stable' && <Minus className="w-4 h-4 text-foreground-muted" />}
          {trendValue && (
            <span className={`text-xs font-medium ${
              trend === 'up' ? 'text-green-400' : 
              trend === 'down' ? 'text-red-400' : 
              'text-foreground-muted'
            }`}>
              {trendValue}
            </span>
          )}
        </div>
      )}
    </div>
  )
}
