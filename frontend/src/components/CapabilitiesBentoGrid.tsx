/**
 * Capabilities Bento Grid - ICU Guardian
 * 
 * High-end SaaS design (Linear, Vercel, Apple style)
 * Glassmorphism + Mouse-tracking spotlight effects
 * Responsive CSS Grid with asymmetric card layout
 */

import { useRef, MouseEvent, useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Activity, ShieldCheck, Moon, Zap, Cable } from 'lucide-react'

// ============================================================================
// ANIMATION VARIANTS
// ============================================================================
const containerVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.2,
      ease: 'easeOut'
    }
  }
}

const cardVariants = {
  hidden: { opacity: 0, scale: 0.9 },
  show: {
    opacity: 1,
    scale: 1,
    transition: {
      duration: 0.6,
      ease: [0.25, 0.46, 0.45, 0.94]
    }
  }
}

// ============================================================================
// SPOTLIGHT CARD COMPONENT - Mouse tracking effect
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
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 })
  const [isHovered, setIsHovered] = useState(false)

  const handleMouseMove = (e: MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return
    const rect = cardRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    setMousePos({ x, y })
  }

  const handleMouseEnter = () => setIsHovered(true)
  const handleMouseLeave = () => setIsHovered(false)

  return (
    <motion.div
      ref={cardRef}
      onMouseMove={handleMouseMove}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      variants={cardVariants}
      className={`
        ${colSpan} ${rowSpan}
        relative group min-h-[200px] rounded-2xl overflow-hidden
        bg-white/[0.05] border border-white/[0.1]
        backdrop-blur-md
        p-6 md:p-8
        transition-all duration-300
        hover:border-white/[0.2] hover:scale-[1.02]
        ${className}
      `}
    >
      {/* Spotlight Gradient - Mouse tracking */}
      {isHovered && (
        <motion.div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: `radial-gradient(
              400px circle at ${mousePos.x}px ${mousePos.y}px,
              rgba(94, 106, 210, 0.15),
              transparent 80%
            )`
          }}
        />
      )}

      {/* Animated Gradient Border on Hover */}
      <div
        className={`
          absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100
          transition-opacity duration-300 pointer-events-none
          bg-gradient-to-br from-accent/20 via-transparent to-transparent
        `}
      />

      {/* Content */}
      <div className="relative z-10 h-full flex flex-col">
        {children}
      </div>
    </motion.div>
  )
}

// ============================================================================
// CARD A - Kinetic Vector Analysis (2x2 Hero Card)
// ============================================================================
function CardA() {
  return (
    <SpotlightCard colSpan="col-span-1 md:col-span-2 lg:col-span-2" rowSpan="row-span-2">
      <div className="flex flex-col h-full space-y-4">
        {/* Icon */}
        <motion.div
          whileHover={{ scale: 1.1, rotate: 5 }}
          className="
            w-14 h-14 rounded-xl
            bg-gradient-to-br from-accent/30 to-accent/10
            border border-accent/40
            flex items-center justify-center
            group/icon
          "
        >
          <Activity className="w-7 h-7 text-accent group-hover/icon:text-accent-bright transition-colors" />
        </motion.div>

        {/* Content */}
        <div className="flex-1">
          <h3 className="text-2xl md:text-3xl font-bold text-white mb-3">
            <span className="bg-gradient-to-b from-white via-white/95 to-white/70 bg-clip-text text-transparent">
              Kinetic Vector Analysis
            </span>
          </h3>
          <p className="text-sm md:text-base text-foreground-muted leading-relaxed">
            Proprietary <span className="text-accent font-semibold">"Jerk Algorithm"</span> differentiates rhythmic breathing from acute seizures using velocity-acceleration vectors.
          </p>
        </div>

        {/* Animated SVG Background Wave */}
        <div className="mt-auto pt-6 relative h-24 overflow-hidden rounded-lg">
          <svg
            className="w-full h-full text-accent/20"
            viewBox="0 0 400 100"
            preserveAspectRatio="none"
          >
            <defs>
              <linearGradient id="waveGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="rgba(94, 106, 210, 0.4)" />
                <stop offset="100%" stopColor="rgba(94, 106, 210, 0.05)" />
              </linearGradient>
            </defs>
            <motion.path
              d="M 0 50 Q 100 25, 200 50 T 400 50 L 400 100 L 0 100 Z"
              fill="url(#waveGradient)"
              animate={{ x: [0, -400, 0] }}
              transition={{ duration: 8, repeat: Infinity, ease: 'linear' }}
            />
            <motion.path
              d="M 0 60 Q 100 35, 200 60 T 400 60 L 400 100 L 0 100 Z"
              fill="rgba(94, 106, 210, 0.1)"
              animate={{ x: [0, -400, 0] }}
              transition={{ duration: 10, repeat: Infinity, ease: 'linear', delay: 0.5 }}
            />
          </svg>
        </div>
      </div>
    </SpotlightCard>
  )
}

// ============================================================================
// CARD B - Adaptive Privacy (Tall Vertical Card)
// ============================================================================
function CardB() {
  return (
    <SpotlightCard colSpan="col-span-1" rowSpan="row-span-2">
      <div className="flex flex-col h-full items-center justify-between space-y-6 text-center">
        {/* Top Content */}
        <div className="space-y-3">
          <h3 className="text-xl md:text-2xl font-bold text-white">
            <span className="bg-gradient-to-b from-white via-white/95 to-white/70 bg-clip-text text-transparent">
              Adaptive Privacy
            </span>
          </h3>
          <p className="text-xs md:text-sm text-foreground-muted leading-relaxed">
            Faces blurred locally. Un-blurring occurs only during confirmed critical alerts.
          </p>
        </div>

        {/* Centered Glowing Shield Icon */}
        <motion.div
          animate={{
            scale: [1, 1.05, 1],
            opacity: [0.7, 1, 0.7]
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: 'easeInOut'
          }}
          className="flex-1 flex items-center justify-center"
        >
          <motion.div
            whileHover={{ scale: 1.15 }}
            className="
              w-20 h-20 md:w-24 md:h-24 rounded-2xl
              bg-gradient-to-br from-accent/40 to-accent/10
              border-2 border-accent/50
              flex items-center justify-center
              group/icon
              shadow-lg shadow-accent/30
            "
          >
            <ShieldCheck className="w-10 h-10 md:w-12 md:h-12 text-accent group-hover/icon:text-accent-bright transition-colors" />
          </motion.div>
        </motion.div>

        {/* HIPAA Badge */}
        <motion.div
          whileHover={{ scale: 1.05 }}
          className="
            px-3 py-1.5 rounded-full text-xs font-mono
            bg-accent/10 border border-accent/30
            text-accent
          "
        >
          HIPAA Compliant
        </motion.div>
      </div>
    </SpotlightCard>
  )
}

// ============================================================================
// CARD C - Zero-Light Tracking
// ============================================================================
function CardC() {
  return (
    <SpotlightCard colSpan="col-span-1" rowSpan="row-span-1">
      <div className="flex flex-col h-full space-y-4">
        {/* Icon */}
        <motion.div
          whileHover={{ scale: 1.1, rotate: -5 }}
          className="
            w-12 h-12 rounded-xl
            bg-gradient-to-br from-accent/30 to-accent/10
            border border-accent/40
            flex items-center justify-center
            group/icon
          "
        >
          <Moon className="w-6 h-6 text-accent group-hover/icon:text-accent-bright transition-colors" />
        </motion.div>

        {/* Content */}
        <div className="flex-1">
          <h3 className="text-lg md:text-xl font-bold text-white mb-2">
            <span className="bg-gradient-to-b from-white via-white/95 to-white/70 bg-clip-text text-transparent">
              Zero-Light Tracking
            </span>
          </h3>
          <p className="text-xs md:text-sm text-foreground-muted">
            99% accuracy during low-light 'sundowning' hours.
          </p>
        </div>

        {/* Stat Badge */}
        <motion.div
          whileHover={{ scale: 1.05 }}
          className="
            px-2 py-1 rounded-lg text-xs font-bold
            bg-accent/10 border border-accent/30
            text-accent
            w-fit
          "
        >
          ✓ 99% Accuracy
        </motion.div>
      </div>
    </SpotlightCard>
  )
}

// ============================================================================
// CARD D - Edge Latency
// ============================================================================
function CardD() {
  return (
    <SpotlightCard colSpan="col-span-1" rowSpan="row-span-1">
      <div className="flex flex-col h-full space-y-4">
        {/* Icon */}
        <motion.div
          whileHover={{ scale: 1.1, rotate: 5 }}
          className="
            w-12 h-12 rounded-xl
            bg-gradient-to-br from-accent/30 to-accent/10
            border border-accent/40
            flex items-center justify-center
            group/icon
          "
        >
          <Zap className="w-6 h-6 text-accent group-hover/icon:text-accent-bright transition-colors" />
        </motion.div>

        {/* Content */}
        <div className="flex-1">
          <h3 className="text-lg md:text-xl font-bold text-white mb-2">
            <span className="bg-gradient-to-b from-white via-white/95 to-white/70 bg-clip-text text-transparent">
              Edge Latency
            </span>
          </h3>
          <p className="text-xs md:text-sm text-foreground-muted">
            &lt; 15ms processing. No cloud lag.
          </p>
        </div>

        {/* Latency Badge */}
        <motion.div
          whileHover={{ scale: 1.05 }}
          className="
            px-2 py-1 rounded-lg text-xs font-bold
            bg-accent/10 border border-accent/30
            text-accent
            w-fit
          "
        >
          Lightning Fast
        </motion.div>
      </div>
    </SpotlightCard>
  )
}

// ============================================================================
// CARD E - Hardware Agnostic (Wide Horizontal)
// ============================================================================
function CardE() {
  const devices = ['CCTV', 'IP Cameras', 'Webcams']

  return (
    <SpotlightCard colSpan="col-span-1 md:col-span-2" rowSpan="row-span-1">
      <div className="flex flex-col md:flex-row h-full gap-6 md:gap-8 items-start md:items-center">
        {/* Icon */}
        <motion.div
          whileHover={{ scale: 1.1, rotate: -5 }}
          className="
            flex-shrink-0 w-14 h-14 rounded-xl
            bg-gradient-to-br from-accent/30 to-accent/10
            border border-accent/40
            flex items-center justify-center
            group/icon
          "
        >
          <Cable className="w-7 h-7 text-accent group-hover/icon:text-accent-bright transition-colors" />
        </motion.div>

        {/* Content */}
        <div className="flex-1 space-y-4">
          <div>
            <h3 className="text-lg md:text-xl font-bold text-white mb-2">
              <span className="bg-gradient-to-b from-white via-white/95 to-white/70 bg-clip-text text-transparent">
                Hardware Agnostic
              </span>
            </h3>
            <p className="text-xs md:text-sm text-foreground-muted">
              Plug &amp; Play. Compatible with existing CCTV, IP Cameras, or standard webcams.
            </p>
          </div>

          {/* Device Tags */}
          <div className="flex flex-wrap gap-2">
            {devices.map((device) => (
              <motion.span
                key={device}
                whileHover={{ scale: 1.05 }}
                className="
                  px-3 py-1 rounded-lg text-xs font-medium
                  bg-white/[0.05] border border-white/[0.1]
                  text-foreground-muted
                  hover:border-accent/40 hover:text-accent
                  transition-colors
                "
              >
                {device}
              </motion.span>
            ))}
          </div>
        </div>
      </div>
    </SpotlightCard>
  )
}

// ============================================================================
// MAIN BENTO GRID SECTION
// ============================================================================
export function CapabilitiesBentoGrid() {
  return (
    <section className="relative py-24 md:py-32 overflow-hidden">
      {/* Gradient line separator */}
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

      {/* Animated background blobs */}
      <motion.div
        animate={{
          opacity: [0.3, 0.5, 0.3],
          scale: [0.9, 1.1, 0.9]
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          ease: 'easeInOut'
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
          ease: 'easeInOut',
          delay: 1
        }}
        className="absolute -bottom-40 -right-40 w-80 h-80 bg-accent/3 rounded-full blur-3xl pointer-events-none"
      />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 32 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
          className="text-center mb-16 md:mb-20"
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
              Capabilities
            </span>
          </motion.span>

          <motion.h2
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.15 }}
            className="text-3xl md:text-4xl lg:text-5xl font-bold tracking-tight mb-4"
          >
            <span className="bg-gradient-to-b from-white via-white/95 to-white/70 bg-clip-text text-transparent">
              Clinical Intelligence Platform
            </span>
          </motion.h2>

          <motion.p
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-base md:text-lg text-foreground-muted max-w-3xl mx-auto"
          >
            Enterprise-grade AI capabilities powered by edge computing and adaptive algorithms
          </motion.p>
        </motion.div>

        {/* Bento Grid */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, amount: 0.2 }}
          className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4 md:gap-6"
        >
          {/* Card A - Kinetic Vector (2x2) */}
          <CardA />

          {/* Card B - Adaptive Privacy (Vertical) */}
          <CardB />

          {/* Card C - Zero-Light */}
          <CardC />

          {/* Card D - Edge Latency */}
          <CardD />

          {/* Card E - Hardware Agnostic (2 wide) */}
          <CardE />
        </motion.div>
      </div>
    </section>
  )
}

export default CapabilitiesBentoGrid
