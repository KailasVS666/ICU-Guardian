/**
 * Secure Core Visualization - ICU Guardian
 * 
 * World-class centerpiece section showcasing Zero-Trust & Edge-Only architecture
 * Features: Floating secure core orb with radial data flow stages
 * Style: Premium dark mode, Apple/Vercel/Linear inspired, glassmorphism
 * Animation: 60fps smooth, scroll-triggered reveals, continuous particle flow
 */

import { useRef, useEffect, useState } from 'react'
import { motion, useInView } from 'framer-motion'
import { Camera, Cpu, Lock, Trash2, Shield, CheckCircle2, Zap } from 'lucide-react'

// ============================================================================
// TYPES & INTERFACES
// ============================================================================
interface Particle {
  id: number
  angle: number
  distance: number
  progress: number
  speed: number
  stage: number // 0-3 for which stage it came from
}

interface FloatingStatProps {
  value: string
  label: string
  delay: number
  position: { top?: string; bottom?: string; left?: string; right?: string }
}

// ============================================================================
// FLOATING STAT COUNTER
// ============================================================================
function FloatingStat({ value, label, delay, position }: FloatingStatProps) {
  const ref = useRef<HTMLDivElement>(null)
  const isInView = useInView(ref, { once: true, amount: 0.5 })
  const [displayValue, setDisplayValue] = useState('0')

  useEffect(() => {
    if (!isInView) return

    const numericMatch = value.match(/\d+/)
    if (!numericMatch) {
      setDisplayValue(value)
      return
    }

    const targetNum = parseInt(numericMatch[0])
    const duration = 2000
    const steps = 60
    const increment = targetNum / steps
    let current = 0

    const timer = setInterval(() => {
      current += increment
      if (current >= targetNum) {
        setDisplayValue(value)
        clearInterval(timer)
      } else {
        const currentStr = Math.floor(current).toString()
        setDisplayValue(value.replace(/\d+/, currentStr))
      }
    }, duration / steps)

    return () => clearInterval(timer)
  }, [isInView, value])

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, scale: 0.8, y: 20 }}
      animate={isInView ? { opacity: 1, scale: 1, y: 0 } : {}}
      transition={{ duration: 0.6, delay, ease: [0.16, 1, 0.3, 1] }}
      className="absolute z-20"
      style={position}
    >
      <div className="
        px-6 py-4 rounded-2xl
        bg-gradient-to-br from-white/[0.08] to-white/[0.02]
        backdrop-blur-xl
        border border-white/[0.12]
        shadow-2xl
      ">
        <div className="text-3xl font-bold bg-gradient-to-b from-accent via-accent/90 to-accent/70 bg-clip-text text-transparent mb-1 font-mono">
          {displayValue}
        </div>
        <div className="text-xs text-foreground-muted uppercase tracking-wider">
          {label}
        </div>
      </div>
    </motion.div>
  )
}

// ============================================================================
// RADIAL STAGE NODE
// ============================================================================
interface StageNodeProps {
  icon: React.ReactNode
  label: string
  description: string
  angle: number
  delay: number
  isInView: boolean
}

function StageNode({ icon, label, description, angle, delay, isInView }: StageNodeProps) {
  const [isHovered, setIsHovered] = useState(false)
  
  // Calculate position on a circle (radius 280px on desktop, 200px on mobile)
  const radius = 280
  const x = Math.cos(angle) * radius
  const y = Math.sin(angle) * radius

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0, x, y }}
      animate={isInView ? { 
        opacity: 1, 
        scale: 1,
        x,
        y
      } : {}}
      transition={{ 
        duration: 0.8, 
        delay: 0.5 + delay,
        ease: [0.16, 1, 0.3, 1]
      }}
      onHoverStart={() => setIsHovered(true)}
      onHoverEnd={() => setIsHovered(false)}
      className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 cursor-pointer group"
      style={{ 
        transform: `translate(calc(-50% + ${x}px), calc(-50% + ${y}px))` 
      }}
    >
      {/* Node Container */}
      <div className="relative">
        {/* Pulsing Glow Ring */}
        <motion.div
          animate={{
            boxShadow: isHovered ? [
              '0 0 0 0 rgba(94, 106, 210, 0.6)',
              '0 0 0 20px rgba(94, 106, 210, 0)',
            ] : [
              '0 0 0 0 rgba(94, 106, 210, 0.4)',
              '0 0 0 16px rgba(94, 106, 210, 0)',
            ]
          }}
          transition={{ duration: 2, repeat: Infinity }}
          className="absolute inset-0 rounded-2xl"
        />

        {/* Main Node */}
        <motion.div
          animate={isHovered ? { scale: 1.1 } : { scale: 1 }}
          className="
            relative w-20 h-20 rounded-2xl
            bg-gradient-to-br from-accent/30 to-accent/10
            border-2 border-accent/50
            backdrop-blur-sm
            flex items-center justify-center
            shadow-xl
          "
        >
          <div className="text-accent">
            {icon}
          </div>
        </motion.div>

        {/* Label */}
        <div className="absolute top-full mt-4 left-1/2 -translate-x-1/2 text-center whitespace-nowrap">
          <div className="text-sm font-semibold text-foreground mb-0.5">{label}</div>
          <div className="text-xs text-foreground-muted">{description}</div>
        </div>

        {/* Hover Tooltip */}
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={isHovered ? { opacity: 1, y: -10 } : { opacity: 0, y: 0 }}
          className="
            absolute bottom-full mb-4 left-1/2 -translate-x-1/2
            px-4 py-2 rounded-lg
            bg-black/90 backdrop-blur-sm
            border border-accent/30
            text-xs text-foreground whitespace-nowrap
            pointer-events-none
          "
        >
          {description}
          <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1 w-2 h-2 rotate-45 bg-black/90 border-r border-b border-accent/30" />
        </motion.div>
      </div>
    </motion.div>
  )
}

// ============================================================================
// SECURE CORE ORB (Center)
// ============================================================================
function SecureCoreOrb({ isInView }: { isInView: boolean }) {
  return (
    <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
      <motion.div
        initial={{ opacity: 0, scale: 0 }}
        animate={isInView ? { opacity: 1, scale: 1 } : {}}
        transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
        className="relative"
      >
        {/* Outer Glow Rings */}
        <motion.div
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.3, 0.6, 0.3]
          }}
          transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
          className="absolute inset-0 w-32 h-32 -m-8 rounded-full bg-accent/20 blur-2xl"
        />
        <motion.div
          animate={{
            scale: [1, 1.15, 1],
            opacity: [0.4, 0.7, 0.4]
          }}
          transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut', delay: 0.5 }}
          className="absolute inset-0 w-28 h-28 -m-6 rounded-full bg-accent/30 blur-xl"
        />

        {/* Main Core Sphere */}
        <div className="
          relative w-32 h-32 rounded-full
          bg-gradient-to-br from-accent/40 via-accent/20 to-transparent
          border border-accent/50
          backdrop-blur-md
          shadow-2xl
          flex items-center justify-center
        ">
          {/* Inner Core */}
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
            className="
              w-20 h-20 rounded-full
              bg-gradient-to-br from-accent/60 to-accent/20
              border-2 border-accent/70
              flex items-center justify-center
            "
          >
            <Shield className="w-10 h-10 text-accent" strokeWidth={1.5} />
          </motion.div>

          {/* Rotating Particles Around Core */}
          {[0, 120, 240].map((_, i) => (
            <motion.div
              key={i}
              animate={{ rotate: 360 }}
              transition={{ duration: 10, repeat: Infinity, ease: 'linear', delay: i * 0.3 }}
              className="absolute inset-0"
            >
              <div 
                className="absolute w-2 h-2 rounded-full bg-accent/80 shadow-lg shadow-accent/50"
                style={{
                  left: '50%',
                  top: '10%',
                  marginLeft: '-4px',
                  marginTop: '-4px'
                }}
              />
            </motion.div>
          ))}
        </div>

        {/* Edge Device Label */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={isInView ? { opacity: 1 } : {}}
          transition={{ delay: 1.2 }}
          className="absolute -bottom-12 left-1/2 -translate-x-1/2 text-center whitespace-nowrap"
        >
          <div className="text-xs font-mono uppercase tracking-widest text-accent mb-1">
            Edge Device
          </div>
          <div className="text-xs text-foreground-muted">Secure Processing Core</div>
        </motion.div>
      </motion.div>
    </div>
  )
}

// ============================================================================
// ANIMATED LIGHT TRAILS (Canvas-based for 60fps performance)
// ============================================================================
function AnimatedLightTrails({ isInView }: { isInView: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const particlesRef = useRef<Particle[]>([])
  const animationRef = useRef<number>()

  // Stage positions (same as StageNode angles)
  const stages = [
    { angle: -Math.PI / 2, color: 'rgba(94, 106, 210, ' },      // Capture (top)
    { angle: 0, color: 'rgba(94, 106, 210, ' },                 // Process (right)
    { angle: Math.PI / 2, color: 'rgba(94, 106, 210, ' },       // Encrypt (bottom)
    { angle: Math.PI, color: 'rgba(94, 106, 210, ' }            // Discard (left)
  ]

  useEffect(() => {
    if (!isInView) return

    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    const updateSize = () => {
      const rect = canvas.getBoundingClientRect()
      canvas.width = rect.width * window.devicePixelRatio
      canvas.height = rect.height * window.devicePixelRatio
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio)
    }
    updateSize()
    window.addEventListener('resize', updateSize)

    // Initialize particles
    const initParticles = () => {
      particlesRef.current = []
      // Create particles continuously
      setInterval(() => {
        if (particlesRef.current.length < 20) {
          const stage = Math.floor(Math.random() * 4)
          particlesRef.current.push({
            id: Math.random(),
            angle: stages[stage].angle,
            distance: 280, // Start from stage node
            progress: 0,
            speed: 0.8 + Math.random() * 0.4,
            stage
          })
        }
      }, 400)
    }

    initParticles()

    // Animation loop
    const animate = () => {
      const rect = canvas.getBoundingClientRect()
      ctx.clearRect(0, 0, rect.width, rect.height)

      const centerX = rect.width / 2
      const centerY = rect.height / 2

      // Update and draw particles
      particlesRef.current = particlesRef.current.filter(particle => {
        particle.progress += 0.01 * particle.speed
        particle.distance = 280 * (1 - particle.progress)

        if (particle.distance <= 50) {
          return false // Remove particle when it reaches core
        }

        const x = centerX + Math.cos(particle.angle) * particle.distance
        const y = centerY + Math.sin(particle.angle) * particle.distance

        // Draw light trail
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, 20)
        const alpha = 0.8 * (1 - particle.progress)
        gradient.addColorStop(0, `${stages[particle.stage].color}${alpha})`)
        gradient.addColorStop(1, `${stages[particle.stage].color}0)`)

        ctx.fillStyle = gradient
        ctx.beginPath()
        ctx.arc(x, y, 8, 0, Math.PI * 2)
        ctx.fill()

        // Draw particle core
        ctx.fillStyle = `${stages[particle.stage].color}${alpha})`
        ctx.beginPath()
        ctx.arc(x, y, 3, 0, Math.PI * 2)
        ctx.fill()

        return true
      })

      animationRef.current = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      window.removeEventListener('resize', updateSize)
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isInView])

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none"
      style={{ width: '100%', height: '100%' }}
    />
  )
}

// ============================================================================
// TRUST BADGES (Inline, below core)
// ============================================================================
function TrustBadges({ isInView }: { isInView: boolean }) {
  const badges = [
    { icon: <Shield className="w-4 h-4" />, label: 'HIPAA Compliant' },
    { icon: <Lock className="w-4 h-4" />, label: 'GDPR Ready' },
    { icon: <CheckCircle2 className="w-4 h-4" />, label: 'AES-256 Encrypted' }
  ]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={isInView ? { opacity: 1, y: 0 } : {}}
      transition={{ delay: 1.5, duration: 0.6 }}
      className="flex items-center justify-center gap-6 flex-wrap"
    >
      {badges.map((badge, i) => (
        <motion.div
          key={badge.label}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={isInView ? { opacity: 1, scale: 1 } : {}}
          transition={{ delay: 1.5 + i * 0.1, duration: 0.4 }}
          whileHover={{ scale: 1.05 }}
          className="
            group flex items-center gap-2 px-4 py-2 rounded-lg
            border border-accent/20
            hover:border-accent/60
            transition-all duration-300
            cursor-default
          "
        >
          <div className="text-accent group-hover:text-accent-bright transition-colors">
            {badge.icon}
          </div>
          <span className="text-xs font-medium text-foreground-muted group-hover:text-foreground transition-colors">
            {badge.label}
          </span>
          <motion.div
            className="h-px w-0 group-hover:w-full bg-accent/60 absolute bottom-0 left-0"
            initial={{ width: 0 }}
            whileHover={{ width: '100%' }}
            transition={{ duration: 0.3 }}
          />
        </motion.div>
      ))}
    </motion.div>
  )
}

// ============================================================================
// MAIN SECURE CORE VISUALIZATION SECTION
// ============================================================================
export function SecureCoreVisualization() {
  const ref = useRef<HTMLDivElement>(null)
  const isInView = useInView(ref, { once: true, amount: 0.3 })

  const stages = [
    {
      icon: <Camera className="w-9 h-9" strokeWidth={1.5} />,
      label: 'Capture',
      description: 'Video streams ingested',
      angle: -Math.PI / 2 // Top
    },
    {
      icon: <Cpu className="w-9 h-9" strokeWidth={1.5} />,
      label: 'Process',
      description: 'AI inference on device',
      angle: 0 // Right
    },
    {
      icon: <Lock className="w-9 h-9" strokeWidth={1.5} />,
      label: 'Encrypt',
      description: 'AES-256 secured alerts',
      angle: Math.PI / 2 // Bottom
    },
    {
      icon: <Trash2 className="w-9 h-9" strokeWidth={1.5} />,
      label: 'Discard',
      description: 'Frames instantly deleted',
      angle: Math.PI // Left
    }
  ]

  return (
    <section ref={ref} className="relative py-32 md:py-40 overflow-hidden">
      {/* Background gradient accent */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-accent/[0.02] to-transparent" />
      
      {/* Top border line */}
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

      <div className="relative max-w-7xl mx-auto px-4">
        {/* Headline */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          className="text-center mb-24"
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={isInView ? { opacity: 1, scale: 1 } : {}}
            transition={{ duration: 0.6 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/[0.03] border border-accent/30 mb-6"
          >
            <Zap className="w-3.5 h-3.5 text-accent" />
            <span className="text-xs font-mono uppercase tracking-widest text-accent">
              Secure by Design
            </span>
          </motion.div>

          <h2 className="text-4xl md:text-5xl lg:text-6xl font-semibold tracking-tight mb-6">
            <span className="bg-gradient-to-b from-white via-white/95 to-white/70 bg-clip-text text-transparent">
              Process & Discard Architecture
            </span>
          </h2>

          <p className="text-lg md:text-xl text-foreground-muted max-w-2xl mx-auto leading-relaxed">
            Intelligence runs locally. Nothing uploaded. Nothing stored.
          </p>
        </motion.div>

        {/* Main Visualization Container */}
        <div className="relative h-[700px] md:h-[800px]">
          {/* Animated Light Trails (Canvas) */}
          <AnimatedLightTrails isInView={isInView} />

          {/* Secure Core Orb (Center) */}
          <SecureCoreOrb isInView={isInView} />

          {/* Radial Stage Nodes */}
          {stages.map((stage, i) => (
            <StageNode
              key={stage.label}
              {...stage}
              delay={i * 0.15}
              isInView={isInView}
            />
          ))}

          {/* Floating Stats */}
          <FloatingStat
            value="15ms"
            label="Processing Latency"
            delay={1.2}
            position={{ top: '10%', right: '5%' }}
          />
          <FloatingStat
            value="0%"
            label="Cloud Dependency"
            delay={1.4}
            position={{ top: '50%', left: '5%' }}
          />
          <FloatingStat
            value="100%"
            label="Local Control"
            delay={1.6}
            position={{ bottom: '15%', right: '8%' }}
          />
        </div>

        {/* Trust Badges */}
        <div className="mt-20 mb-16">
          <TrustBadges isInView={isInView} />
        </div>

        {/* Micro-copy */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={isInView ? { opacity: 1 } : {}}
          transition={{ delay: 2, duration: 1 }}
          className="text-center max-w-3xl mx-auto"
        >
          <p className="text-base md:text-lg text-foreground leading-relaxed font-light tracking-wide">
            All intelligence runs locally.<br />
            Nothing is uploaded.<br />
            Nothing is stored.<br />
            Nothing leaves the room.
          </p>
        </motion.div>
      </div>

      {/* Bottom border line */}
      <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />
    </section>
  )
}

export default SecureCoreVisualization
