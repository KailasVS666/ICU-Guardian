import { useRef, MouseEvent, ReactNode } from 'react'
import { AlertCircle, Activity, Heart, TrendingDown, Shield, Zap } from 'lucide-react'

interface PillarStatusProps {
  alerts: any
}

interface PillarConfig {
  icon: ReactNode
  color: string
  bgColor: string
  borderColor: string
  glowColor: string
  label: string
}

const pillarConfig: { [key: string]: PillarConfig } = {
  self_harm: { 
    icon: <AlertCircle className="w-5 h-5" />, 
    color: 'text-red-400',
    bgColor: 'bg-red-500/10',
    borderColor: 'border-red-500/30',
    glowColor: 'rgba(239,68,68,0.1)',
    label: 'Self Harm'
  },
  agitation: { 
    icon: <Activity className="w-5 h-5" />, 
    color: 'text-orange-400',
    bgColor: 'bg-orange-500/10',
    borderColor: 'border-orange-500/30',
    glowColor: 'rgba(249,115,22,0.1)',
    label: 'Agitation'
  },
  distress: { 
    icon: <Heart className="w-5 h-5" />, 
    color: 'text-yellow-400',
    bgColor: 'bg-yellow-500/10',
    borderColor: 'border-yellow-500/30',
    glowColor: 'rgba(234,179,8,0.1)',
    label: 'Distress'
  },
  fall: { 
    icon: <TrendingDown className="w-5 h-5" />, 
    color: 'text-violet-400',
    bgColor: 'bg-violet-500/10',
    borderColor: 'border-violet-500/30',
    glowColor: 'rgba(139,92,246,0.1)',
    label: 'Fall Risk'
  },
}

function PillarCard({ 
  name: _name, 
  data, 
  config 
}: { 
  name: string
  data: any
  config: PillarConfig 
}) {
  const cardRef = useRef<HTMLDivElement>(null)
  const isActive = data?.active

  const handleMouseMove = (e: MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return
    const rect = cardRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    cardRef.current.style.setProperty('--mouse-x', `${x}px`)
    cardRef.current.style.setProperty('--mouse-y', `${y}px`)
  }

  return (
    <div
      ref={cardRef}
      onMouseMove={handleMouseMove}
      className={`
        relative p-4 rounded-xl border overflow-hidden
        transition-all duration-300 ease-expo-out
        hover:transform hover:-translate-y-1
        ${isActive 
          ? `${config.bgColor} ${config.borderColor}` 
          : 'bg-white/[0.02] border-white/[0.06]'
        }
      `}
      style={{
        boxShadow: isActive 
          ? `0 0 0 1px ${config.glowColor}, 0 8px 30px ${config.glowColor}` 
          : undefined
      }}
    >
      {/* Spotlight effect */}
      <div 
        className="absolute inset-0 opacity-0 hover:opacity-100 transition-opacity duration-300 pointer-events-none"
        style={{
          background: `radial-gradient(200px circle at var(--mouse-x, 50%) var(--mouse-y, 50%), ${config.glowColor}, transparent 70%)`
        }}
      />
      
      {/* Header */}
      <div className="flex items-center justify-between mb-3 relative z-10">
        <div className={`p-2 rounded-lg ${isActive ? config.bgColor : 'bg-white/[0.03]'} border ${isActive ? config.borderColor : 'border-white/[0.06]'}`}>
          <span className={isActive ? config.color : 'text-foreground-muted'}>
            {config.icon}
          </span>
        </div>
        
        {isActive && (
          <span className={`
            inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full 
            text-[10px] font-semibold uppercase tracking-wider
            ${config.bgColor} ${config.color} border ${config.borderColor}
          `}>
            <span className="relative flex h-1.5 w-1.5">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${config.color.replace('text-', 'bg-')}`} />
              <span className={`relative inline-flex rounded-full h-1.5 w-1.5 ${config.color.replace('text-', 'bg-')}`} />
            </span>
            Active
          </span>
        )}
      </div>
      
      {/* Title */}
      <h4 className={`font-semibold text-sm mb-1 relative z-10 ${isActive ? 'text-foreground' : 'text-foreground-muted'}`}>
        {config.label}
      </h4>
      
      {/* Alert count */}
      <p className="text-xs text-foreground-muted relative z-10">
        <span className={`font-mono font-semibold ${isActive ? config.color : ''}`}>
          {data?.count || 0}
        </span>
        {' '}alert{(data?.count || 0) !== 1 ? 's' : ''} recorded
      </p>
    </div>
  )
}

export default function PillarStatus({ alerts }: PillarStatusProps) {
  const pillars = alerts?.pillars || {}
  const cardRef = useRef<HTMLDivElement>(null)

  const handleMouseMove = (e: MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return
    const rect = cardRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    cardRef.current.style.setProperty('--mouse-x', `${x}px`)
    cardRef.current.style.setProperty('--mouse-y', `${y}px`)
  }

  return (
    <div 
      ref={cardRef}
      onMouseMove={handleMouseMove}
      className="glass-card spotlight-card p-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-accent/10 border border-accent/20">
            <Shield className="w-5 h-5 text-accent" />
          </div>
          <div>
            <h3 className="text-lg font-semibold tracking-tight text-foreground">
              Detection Pillars
            </h3>
            <p className="text-xs text-foreground-muted">
              AI-powered monitoring systems
            </p>
          </div>
        </div>
        
        {/* Active count */}
        <div className="flex items-center gap-2 px-3 py-1.5 bg-white/[0.03] rounded-lg border border-white/[0.06]">
          <Zap className="w-4 h-4 text-accent" />
          <span className="text-xs font-mono text-foreground-muted">
            {Object.values(pillars).filter((p: any) => p?.active).length} active
          </span>
        </div>
      </div>
      
      {/* Pillar Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {Object.entries(pillarConfig).map(([name, config]) => (
          <PillarCard
            key={name}
            name={name}
            data={pillars[name]}
            config={config}
          />
        ))}
      </div>
    </div>
  )
}
