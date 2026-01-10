/**
 * Bento Features Grid - ICU Guardian
 * 
 * Apple-style Bento cards using Glassmorphism
 * Responsive CSS Grid with asymmetric layout
 * Mouse-tracking spotlight effects on hover
 * Scroll-triggered reveal animations (Apple website style)
 */

import { useRef, MouseEvent } from 'react'
import { motion, useInView } from 'framer-motion'
import { 
  Activity, 
  ShieldCheck, 
  Moon, 
  Zap, 
  Cable
} from 'lucide-react'

// ============================================================================
// SCROLL-REVEAL CARD WRAPPER - Apple-style animation
// ============================================================================
interface ScrollRevealCardProps {
  children: React.ReactNode
  colSpan?: string
  rowSpan?: string
  delay?: number
}

function ScrollRevealCard({ 
  children, 
  colSpan = 'col-span-1',
  rowSpan = 'row-span-1',
  delay = 0
}: ScrollRevealCardProps) {
  const ref = useRef<HTMLDivElement>(null)
  const isInView = useInView(ref, { 
    once: true,
    amount: 0.2,
    margin: "0px 0px -100px 0px"
  })

  return (
    <motion.div
      ref={ref}
      className={`${colSpan} ${rowSpan}`}
      initial={{ opacity: 0, y: 40, scale: 0.95 }}
      animate={isInView ? { 
        opacity: 1, 
        y: 0, 
        scale: 1 
      } : { 
        opacity: 0, 
        y: 40, 
        scale: 0.95 
      }}
      transition={{
        duration: 0.7,
        delay: delay * 0.1,
        ease: [0.25, 0.46, 0.45, 0.94] // Smooth cubic-bezier
      }}
    >
      {children}
    </motion.div>
  )
}

// ============================================================================
interface SpotlightCardProps {
  children: React.ReactNode
  className?: string
  colSpan?: string
  rowSpan?: string
}

function SpotlightCard({ 
  children, 
  className = '',
  colSpan = 'col-span-1',
  rowSpan = 'row-span-1'
}: SpotlightCardProps) {
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
      className={`
        ${colSpan} ${rowSpan}
        relative group rounded-3xl overflow-hidden
        bg-gradient-to-b from-white/[0.08] to-white/[0.02]
        border border-white/[0.06] hover:border-accent/40
        backdrop-blur-xl
        p-8 md:p-10 lg:p-12
        transition-all duration-300
        hover:shadow-[0_0_0_1px_rgba(94,106,210,0.3),0_8px_40px_rgba(0,0,0,0.5),0_0_80px_rgba(94,106,210,0.15)]
        ${className}
      `}
      style={{
        // @ts-ignore
        '--mouse-x': '0px',
        '--mouse-y': '0px',
      } as React.CSSProperties}
    >
      {/* Mouse-tracking spotlight gradient */}
      <div
        className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
        style={{
          // @ts-ignore
          background: `radial-gradient(300px circle at var(--mouse-x) var(--mouse-y), rgba(94, 106, 210, 0.2), transparent 80%)`,
        }}
      />

      {/* Content */}
      <div className="relative z-10 flex flex-col h-full">
        {children}
      </div>
    </div>
  )
}

// ============================================================================
// CARD A - Kinetic Vector Analysis (Large - Top Left)
// ============================================================================
function KineticVectorCard() {
  return (
    <SpotlightCard colSpan="md:col-span-2 col-span-1" rowSpan="md:row-span-2 row-span-1">
      <div className="space-y-6 h-full flex flex-col">
        {/* Icon with animated background */}
        <motion.div 
          whileHover={{ scale: 1.08, rotate: 2 }}
          className="w-16 h-16 rounded-2xl bg-gradient-to-br from-accent/30 to-accent/10 border-2 border-accent/50 flex items-center justify-center group/icon shadow-lg shadow-accent/20"
        >
          <Activity className="w-8 h-8 text-accent group-hover/icon:text-accent-bright transition-colors" />
        </motion.div>

        {/* Title */}
        <div className="space-y-3">
          <h3 className="text-3xl md:text-4xl font-bold text-foreground leading-tight">
            <span className="bg-gradient-to-b from-white via-white/95 to-white/70 bg-clip-text text-transparent">
              Kinetic Vector Dynamics
            </span>
          </h3>
          <p className="text-base text-foreground-muted leading-relaxed">
            Proprietary algorithm distinguishes between rhythmic respiratory motion and acute agitation events using velocity-acceleration vectors. Filters 99% of false alarms.
          </p>
        </div>

        {/* Visual: Abstract medical data wave pattern */}
        <div className="mt-auto pt-8 relative h-40 overflow-hidden rounded-xl bg-gradient-to-b from-accent/10 to-transparent">
          <svg 
            className="w-full h-full text-accent/30"
            viewBox="0 0 400 200" 
            preserveAspectRatio="none"
          >
            {/* Multiple wave layers */}
            <path
              d="M 0 100 Q 100 50, 200 100 T 400 100"
              stroke="currentColor"
              strokeWidth="2"
              fill="none"
              className="animate-pulse"
            />
            <path
              d="M 0 120 Q 100 70, 200 120 T 400 120"
              stroke="currentColor"
              strokeWidth="1.5"
              fill="none"
              opacity="0.6"
            />
            <path
              d="M 0 80 Q 100 30, 200 80 T 400 80"
              stroke="currentColor"
              strokeWidth="1"
              fill="none"
              opacity="0.4"
            />
            {/* Grid lines */}
            <line x1="0" y1="50" x2="400" y2="50" stroke="currentColor" strokeWidth="0.5" opacity="0.3" />
            <line x1="0" y1="100" x2="400" y2="100" stroke="currentColor" strokeWidth="0.5" opacity="0.3" />
            <line x1="0" y1="150" x2="400" y2="150" stroke="currentColor" strokeWidth="0.5" opacity="0.3" />
          </svg>
          <div className="absolute inset-0 bg-gradient-to-t from-background-base/80 to-transparent pointer-events-none" />
        </div>

        {/* Stat badge */}
        <div className="pt-4 border-t border-white/[0.06]">
          <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-mono bg-accent/10 border border-accent/30 text-accent">
            ✓ 99% Accuracy
          </span>
        </div>
      </div>
    </SpotlightCard>
  )
}

// ============================================================================
// CARD B - Adaptive Privacy (Tall - Top Right)
// ============================================================================
function AdaptivePrivacyCard() {
  return (
    <SpotlightCard colSpan="md:col-span-1 col-span-1" rowSpan="md:row-span-2 row-span-1">
      <div className="space-y-6 flex flex-col h-full">
        {/* Icon */}
        <motion.div 
          whileHover={{ scale: 1.08, rotate: -2 }}
          className="w-16 h-16 rounded-2xl bg-gradient-to-br from-accent/30 to-accent/10 border-2 border-accent/50 flex items-center justify-center group/icon shadow-lg shadow-accent/20"
        >
          <ShieldCheck className="w-8 h-8 text-accent group-hover/icon:text-accent-bright transition-colors" />
        </motion.div>

        {/* Title */}
        <div className="space-y-3">
          <h3 className="text-2xl md:text-3xl font-bold text-foreground leading-tight">
            <span className="bg-gradient-to-b from-white via-white/95 to-white/70 bg-clip-text text-transparent">
              HIPAA-Grade Privacy
            </span>
          </h3>
          <p className="text-sm text-foreground-muted leading-relaxed">
            Patient faces are blurred locally on the edge device. The system only un-blurs video streams during confirmed critical alerts to verify safety.
          </p>
        </div>

        {/* Trust badges */}
        <div className="mt-auto space-y-3 pt-6 border-t border-white/[0.06]">
          <p className="text-xs font-mono uppercase tracking-widest text-accent/70 font-bold">Compliance Standards</p>
          <div className="space-y-2">
            {[
              { label: 'HIPAA', desc: 'Healthcare Privacy' },
              { label: 'GDPR', desc: 'Data Protection' },
              { label: 'CCPA', desc: 'Consumer Privacy' }
            ].map((badge) => (
              <motion.div
                key={badge.label}
                whileHover={{ x: 4 }}
                className="px-3 py-2 rounded-lg bg-gradient-to-r from-accent/5 to-transparent border border-accent/20 hover:border-accent/40 transition-all"
              >
                <p className="text-sm font-semibold text-accent">{badge.label}</p>
                <p className="text-xs text-foreground-muted">{badge.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </SpotlightCard>
  )
}

// ============================================================================
// CARD C - Zero-Light Tracking (Standard - Middle Right)
// ============================================================================
function ZeroLightTrackingCard() {
  return (
    <SpotlightCard colSpan="md:col-span-1 col-span-1" rowSpan="row-span-1">
      <div className="space-y-4 h-full">
        {/* Icon */}
        <motion.div 
          whileHover={{ scale: 1.08, rotate: 2 }}
          className="w-14 h-14 rounded-2xl bg-gradient-to-br from-accent/30 to-accent/10 border-2 border-accent/50 flex items-center justify-center group/icon shadow-lg shadow-accent/20"
        >
          <Moon className="w-7 h-7 text-accent group-hover/icon:text-accent-bright transition-colors" />
        </motion.div>

        {/* Content */}
        <div className="space-y-2">
          <h3 className="text-2xl font-bold text-foreground">
            <span className="bg-gradient-to-b from-white via-white/95 to-white/70 bg-clip-text text-transparent">
              Night Vision
            </span>
          </h3>
          <p className="text-sm text-foreground-muted leading-relaxed">
            Contrast-boosting algorithms ensure accuracy during 'sundowning' hours.
          </p>
        </div>
      </div>
    </SpotlightCard>
  )
}

// ============================================================================
// CARD D - Edge Latency (Standard - Bottom Right)
// ============================================================================
function EdgeLatencyCard() {
  return (
    <SpotlightCard colSpan="md:col-span-1 col-span-1" rowSpan="row-span-1">
      <div className="space-y-4 h-full">
        {/* Icon */}
        <motion.div 
          whileHover={{ scale: 1.08, rotate: -2 }}
          className="w-14 h-14 rounded-2xl bg-gradient-to-br from-accent/30 to-accent/10 border-2 border-accent/50 flex items-center justify-center group/icon shadow-lg shadow-accent/20"
        >
          <Zap className="w-7 h-7 text-accent group-hover/icon:text-accent-bright transition-colors" />
        </motion.div>

        {/* Content */}
        <div className="space-y-2">
          <h3 className="text-2xl font-bold text-foreground">
            <span className="bg-gradient-to-b from-white via-white/95 to-white/70 bg-clip-text text-transparent">
              &lt; 15ms Latency
            </span>
          </h3>
          <p className="text-sm text-foreground-muted leading-relaxed">
            No cloud lag. Instant local processing.
          </p>
        </div>
      </div>
    </SpotlightCard>
  )
}

// ============================================================================
// CARD E - Hardware Agnostic (Wide - Bottom Left)
// ============================================================================
function HardwareAgnosticCard() {
  return (
    <SpotlightCard colSpan="md:col-span-2 col-span-1" rowSpan="row-span-1">
      <div className="space-y-5 flex flex-col md:flex-row items-start md:items-center gap-8">
        {/* Icon */}
        <motion.div 
          whileHover={{ scale: 1.08, rotate: 2 }}
          className="w-16 h-16 rounded-2xl bg-gradient-to-br from-accent/30 to-accent/10 border-2 border-accent/50 flex items-center justify-center group/icon flex-shrink-0 shadow-lg shadow-accent/20"
        >
          <Cable className="w-8 h-8 text-accent group-hover/icon:text-accent-bright transition-colors" />
        </motion.div>

        {/* Content */}
        <div className="flex-1 space-y-3">
          <h3 className="text-2xl font-bold text-foreground">
            <span className="bg-gradient-to-b from-white via-white/95 to-white/70 bg-clip-text text-transparent">
              Plug-and-Play Infrastructure
            </span>
          </h3>
          <p className="text-sm text-foreground-muted leading-relaxed">
            Compatible with existing CCTV, IP Cameras, or standard webcams. No expensive LIDAR required.
          </p>
          <div className="flex flex-wrap gap-2 pt-2">
            {['CCTV', 'IP Cameras', 'Webcams', 'USB Devices'].map((item) => (
              <span 
                key={item}
                className="px-3 py-1 rounded-lg text-xs font-medium bg-white/[0.03] border border-white/[0.08] text-foreground-muted"
              >
                {item}
              </span>
            ))}
          </div>
        </div>
      </div>
    </SpotlightCard>
  )
}

// ============================================================================
// BENTO GRID CONTAINER
// ============================================================================
export function BentoFeaturesGrid() {
  return (
    <section id="features-bento" className="relative py-24 md:py-40 overflow-hidden">
      {/* Top gradient line */}
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

      {/* Animated background blobs for visual interest */}
      <motion.div
        animate={{ 
          opacity: [0.3, 0.5, 0.3],
          scale: [0.9, 1.1, 0.9]
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          ease: "easeInOut"
        }}
        className="absolute -top-40 -left-40 w-96 h-96 bg-accent/5 rounded-full blur-3xl pointer-events-none"
      />
      <motion.div
        animate={{ 
          opacity: [0.2, 0.4, 0.2],
          scale: [1, 0.9, 1]
        }}
        transition={{
          duration: 12,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 1
        }}
        className="absolute -bottom-40 -right-40 w-80 h-80 bg-accent/3 rounded-full blur-3xl pointer-events-none"
      />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Section Header with enhanced styling */}
        <motion.div
          initial={{ opacity: 0, y: 32 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
          className="mb-20 text-center"
        >
          <motion.span 
            initial={{ opacity: 0, scale: 0.8 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-accent/5 border border-accent/20 mb-6"
          >
            <div className="w-2 h-2 rounded-full bg-accent/60 animate-pulse" />
            <span className="text-xs font-mono uppercase tracking-widest text-accent">
              Core Features
            </span>
          </motion.span>

          <motion.h2 
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.15 }}
            className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight mb-6"
          >
            <span className="bg-gradient-to-b from-white via-white/95 to-white/70 bg-clip-text text-transparent">
              Enterprise-Grade
            </span>
            <br />
            <span className="bg-gradient-to-r from-accent via-indigo-400 to-accent bg-clip-text text-transparent animate-pulse">
              Clinical Intelligence
            </span>
          </motion.h2>

          <motion.p 
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-lg text-foreground-muted max-w-3xl mx-auto leading-relaxed"
          >
            Proprietary AI system combining kinetic vector analysis, adaptive privacy controls, and edge-first architecture for real-time patient safety monitoring.
          </motion.p>
        </motion.div>

        {/* Bento Grid - Asymmetric layout with enhanced spacing */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.25 }}
          className="grid grid-cols-1 md:grid-cols-4 gap-6 auto-rows-[auto]"
        >
          {/* Card A - Kinetic Vector (2x2 on desktop, 1x1 on mobile) */}
          <ScrollRevealCard colSpan="md:col-span-2 col-span-1" rowSpan="md:row-span-2 row-span-1" delay={0}>
            <KineticVectorCard />
          </ScrollRevealCard>

          {/* Card B - Adaptive Privacy (1x2 on desktop, 1x1 on mobile) */}
          <ScrollRevealCard colSpan="md:col-span-1 col-span-1" rowSpan="md:row-span-2 row-span-1" delay={1}>
            <AdaptivePrivacyCard />
          </ScrollRevealCard>

          {/* Card C - Zero-Light Tracking (1x1) */}
          <ScrollRevealCard colSpan="md:col-span-1 col-span-1" rowSpan="row-span-1" delay={2}>
            <ZeroLightTrackingCard />
          </ScrollRevealCard>

          {/* Card D - Edge Latency (1x1) */}
          <ScrollRevealCard colSpan="md:col-span-1 col-span-1" rowSpan="row-span-1" delay={3}>
            <EdgeLatencyCard />
          </ScrollRevealCard>

          {/* Card E - Hardware Agnostic (2x1 on desktop, 1x1 on mobile) */}
          <ScrollRevealCard colSpan="md:col-span-2 col-span-1" rowSpan="row-span-1" delay={4}>
            <HardwareAgnosticCard />
          </ScrollRevealCard>
        </motion.div>
      </div>
    </section>
  )
}

export default BentoFeaturesGrid
