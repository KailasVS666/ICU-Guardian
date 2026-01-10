/**
 * Impact by the Numbers Section - ICU Guardian
 * 
 * 4-Column Statistic Grid
 * Subtle grid pattern background
 * Framer-motion countup animations on scroll
 */

import { useRef, useEffect, useState } from 'react'
import { motion, useInView } from 'framer-motion'

// ============================================================================
// COUNTUP COMPONENT - Animate numbers on scroll
// ============================================================================
interface CountUpProps {
  end: number
  suffix?: string
  prefix?: string
  duration?: number
  delay?: number
}

function CountUpNumber({ end, suffix = '', prefix = '', duration = 2.5, delay = 0 }: CountUpProps) {
  const [count, setCount] = useState(0)
  const ref = useRef<HTMLDivElement>(null)
  const isInView = useInView(ref, { once: true, margin: "-100px" })

  useEffect(() => {
    if (!isInView) return

    const startTime = Date.now()
    const durationMs = duration * 1000

    const animate = () => {
      const now = Date.now()
      const elapsed = now - startTime
      const progress = Math.min(elapsed / durationMs, 1)

      // Easing function for smoother animation
      const easeOutQuad = (t: number) => t * (2 - t)
      const easedProgress = easeOutQuad(progress)

      setCount(Math.floor(end * easedProgress))

      if (progress < 1) {
        requestAnimationFrame(animate)
      } else {
        setCount(end)
      }
    }

    // Add delay before starting animation
    const timeoutId = setTimeout(() => {
      animate()
    }, delay * 1000)

    return () => clearTimeout(timeoutId)
  }, [isInView, end, duration, delay])

  return (
    <div ref={ref} className="text-foreground">
      {prefix}
      {count}
      {suffix}
    </div>
  )
}

// ============================================================================
// STATISTIC CARD
// ============================================================================
interface StatisticCardProps {
  number: number
  suffix?: string
  prefix?: string
  label: string
  description?: string
  delay: number
}

function StatisticCard({ 
  number, 
  suffix = '', 
  prefix = '',
  label, 
  description,
  delay 
}: StatisticCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay }}
      className="
        relative group
        p-6 md:p-8
        rounded-2xl
        bg-gradient-to-b from-white/[0.05] to-white/[0.01]
        border border-white/[0.06]
        hover:border-white/[0.12]
        backdrop-blur-xl
        transition-all duration-300
        hover:shadow-[0_0_0_1px_rgba(255,255,255,0.1),0_4px_30px_rgba(0,0,0,0.3)]
      "
    >
      {/* Animated glow on hover */}
      <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
        style={{
          background: 'radial-gradient(circle at center, rgba(94, 106, 210, 0.1), transparent)',
          filter: 'blur(40px)'
        }}
      />

      {/* Content */}
      <div className="relative z-10 space-y-4">
        {/* Big Number */}
        <div className="text-4xl md:text-5xl lg:text-6xl font-bold text-white">
          <CountUpNumber 
            end={number}
            suffix={suffix}
            prefix={prefix}
            duration={2.5}
            delay={delay}
          />
        </div>

        {/* Label - Teal */}
        <div className="space-y-2">
          <p className="text-sm font-mono uppercase tracking-widest text-accent font-semibold">
            {label}
          </p>
          {description && (
            <p className="text-sm text-foreground-muted">
              {description}
            </p>
          )}
        </div>
      </div>
    </motion.div>
  )
}

// ============================================================================
// MAIN IMPACT STATISTICS SECTION
// ============================================================================
export function ImpactStatistics() {
  const statistics = [
    {
      number: 15,
      suffix: 'ms',
      label: 'Edge Processing Latency',
      description: 'Real-time video analysis without cloud lag',
    },
    {
      number: 94,
      suffix: '%',
      label: 'Reduction in False Alarms',
      description: 'Kinetic vector analysis accuracy',
    },
    {
      number: 24,
      suffix: '/7',
      label: 'Autonomous Monitoring',
      description: 'Continuous 24/7/365 vigilance',
    },
    {
      number: 0.1,
      suffix: '%',
      label: 'System Downtime',
      description: 'Enterprise-grade reliability',
    },
  ]

  return (
    <section id="impact" className="relative py-24 md:py-32 overflow-hidden">
      {/* Top gradient line */}
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

      {/* Grid Pattern Background */}
      <div className="absolute inset-0 opacity-[0.03]">
        <div 
          style={{
            backgroundImage: `
              linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px),
              linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px)
            `,
            backgroundSize: '80px 80px'
          }}
        />
      </div>

      {/* Animated gradient blob background */}
      <motion.div
        animate={{
          opacity: [0.3, 0.5, 0.3],
          scale: [1, 1.1, 1],
        }}
        transition={{
          duration: 8,
          repeat: Infinity,
          ease: "easeInOut"
        }}
        className="absolute -bottom-1/4 -right-1/4 w-96 h-96 bg-accent/10 rounded-full blur-3xl pointer-events-none"
      />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          className="text-center mb-16"
        >
          <span className="inline-block text-xs font-mono uppercase tracking-widest text-accent mb-4">
            Social Proof
          </span>
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-semibold tracking-tight">
            <span className="bg-gradient-to-b from-white via-white/95 to-white/70 bg-clip-text text-transparent">
              Impact by the Numbers
            </span>
          </h2>
          <p className="text-lg text-foreground-muted max-w-2xl mx-auto mt-4">
            Measurable outcomes that prove ICU Guardian's clinical effectiveness and operational efficiency.
          </p>
        </motion.div>

        {/* Statistics Grid - 4 columns on desktop, 1-2 on mobile/tablet */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {statistics.map((stat, index) => (
            <StatisticCard 
              key={stat.label}
              {...stat}
              delay={index * 0.1}
            />
          ))}
        </div>

        {/* Trust Statement */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="
            mt-16 p-8 rounded-2xl
            bg-gradient-to-r from-accent/5 to-accent/[0.02]
            border border-accent/20
            text-center
          "
        >
          <p className="text-foreground-muted">
            These metrics are validated through real-world ICU deployments and continuous monitoring systems. 
            <span className="text-accent font-semibold"> Verified results, not marketing claims.</span>
          </p>
        </motion.div>
      </div>
    </section>
  )
}

export default ImpactStatistics
