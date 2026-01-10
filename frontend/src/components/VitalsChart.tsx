import { useState, useEffect, useRef, MouseEvent } from 'react'
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Area, AreaChart } from 'recharts'
import { Activity, TrendingUp } from 'lucide-react'

interface VitalsChartProps {
  vitals: any
}

// Custom tooltip matching design system
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass-card p-3 border border-white/10 backdrop-blur-xl">
        <p className="text-xs font-mono text-foreground-muted mb-2">{label}</p>
        {payload.map((entry: any, index: number) => (
          <div key={index} className="flex items-center gap-2">
            <div 
              className="w-2 h-2 rounded-full" 
              style={{ backgroundColor: entry.color }}
            />
            <span className="text-sm text-foreground">
              {entry.name}: <span className="font-semibold">{entry.value?.toFixed(1)}</span>
            </span>
          </div>
        ))}
      </div>
    )
  }
  return null
}

export default function VitalsChart({ vitals }: VitalsChartProps) {
  const [data, setData] = useState<any[]>([])
  const cardRef = useRef<HTMLDivElement>(null)

  const handleMouseMove = (e: MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return
    const rect = cardRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    cardRef.current.style.setProperty('--mouse-x', `${x}px`)
    cardRef.current.style.setProperty('--mouse-y', `${y}px`)
  }

  useEffect(() => {
    if (vitals) {
      const newPoint = {
        time: new Date().toLocaleTimeString('en-US', { 
          hour12: false, 
          hour: '2-digit', 
          minute: '2-digit',
          second: '2-digit'
        }),
        hr: vitals.vitals.HR_Avg,
        spo2: vitals.vitals.SpO2_Min,
        timestamp: Date.now(),
      }

      setData((prev) => {
        const updated = [...prev, newPoint]
        return updated.slice(-60)
      })
    }
  }, [vitals])

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
            <Activity className="w-5 h-5 text-accent" />
          </div>
          <div>
            <h3 className="text-lg font-semibold tracking-tight text-foreground">
              Live Vitals Monitoring
            </h3>
            <p className="text-xs text-foreground-muted">
              Real-time patient metrics
            </p>
          </div>
        </div>
        
        {/* Data point indicator */}
        <div className="flex items-center gap-2 px-3 py-1.5 bg-white/[0.03] rounded-lg border border-white/[0.06]">
          <TrendingUp className="w-4 h-4 text-accent" />
          <span className="text-xs font-mono text-foreground-muted">
            {data.length} points
          </span>
        </div>
      </div>
      
      {/* Chart */}
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              {/* Heart rate gradient */}
              <linearGradient id="hrGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#ef4444" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#ef4444" stopOpacity={0} />
              </linearGradient>
              {/* SpO2 gradient */}
              <linearGradient id="spo2Gradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#5E6AD2" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#5E6AD2" stopOpacity={0} />
              </linearGradient>
            </defs>
            
            <CartesianGrid 
              strokeDasharray="3 3" 
              stroke="rgba(255,255,255,0.03)" 
              vertical={false}
            />
            <XAxis 
              dataKey="time" 
              stroke="#8A8F98"
              tick={{ fontSize: 11, fill: '#8A8F98' }}
              tickLine={{ stroke: 'rgba(255,255,255,0.06)' }}
              axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
            />
            <YAxis 
              stroke="#8A8F98" 
              tick={{ fontSize: 11, fill: '#8A8F98' }}
              tickLine={{ stroke: 'rgba(255,255,255,0.06)' }}
              axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend 
              wrapperStyle={{ 
                paddingTop: '16px',
                fontSize: '12px'
              }}
              iconType="circle"
              iconSize={8}
            />
            
            {/* Heart Rate Area */}
            <Area 
              type="monotone" 
              dataKey="hr" 
              stroke="#ef4444" 
              strokeWidth={2}
              fill="url(#hrGradient)"
              name="Heart Rate"
              dot={false}
              activeDot={{ 
                r: 4, 
                strokeWidth: 2, 
                stroke: '#ef4444',
                fill: '#050506'
              }}
            />
            
            {/* SpO2 Area */}
            <Area 
              type="monotone" 
              dataKey="spo2" 
              stroke="#5E6AD2" 
              strokeWidth={2}
              fill="url(#spo2Gradient)"
              name="SpO2"
              dot={false}
              activeDot={{ 
                r: 4, 
                strokeWidth: 2, 
                stroke: '#5E6AD2',
                fill: '#050506'
              }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
