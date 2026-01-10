/**
 * Deployment Roadmap Section - ICU Guardian
 * 
 * Professional Interactive Timeline
 * Hover-based expansion with smooth transitions
 * Clean, minimal deployment process visualization
 */

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Plug, Zap, Shield, ChevronRight, Clock } from 'lucide-react'

// ============================================================================
// INTERACTIVE STEP CARD COMPONENT
// ============================================================================
interface StepCardProps {
  step: number
  icon: React.ReactNode
  title: string
  description: string
  duration: string
  details: string[]
  delay: number
  isLast: boolean
}

function InteractiveStepCard({ 
  step, 
  icon, 
  title, 
  description, 
  duration,
  details,
  delay, 
  isLast
}: StepCardProps) {
  const [isExpanded, setIsExpanded] = useState(false)

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay }}
      onHoverStart={() => setIsExpanded(true)}
      onHoverEnd={() => setIsExpanded(false)}
      className="flex flex-col items-center relative"
    >
      {/* Step Circle */}
      <motion.div
        whileHover={{ scale: 1.05 }}
        className="
          relative z-10 w-16 h-16 rounded-full
          bg-gradient-to-br from-accent/20 to-accent/5
          border-2 border-accent/50
          flex items-center justify-center
          mb-6
          transition-all duration-300
          cursor-pointer
        "
      >
        <motion.div
          animate={isExpanded ? { scale: 1.1 } : { scale: 1 }}
          className="text-accent text-xl font-bold"
        >
          {step}
        </motion.div>

        {/* Subtle hover glow */}
        <motion.div
          animate={{ 
            opacity: isExpanded ? 1 : 0,
            scale: isExpanded ? 1 : 0.8
          }}
          className="absolute -inset-2 rounded-full bg-accent/10 -z-10 blur-lg"
        />
      </motion.div>

      {/* Floating Icon */}
      <motion.div
        initial={{ scale: 0, y: 20 }}
        whileInView={{ scale: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.4, delay: delay + 0.1, type: 'spring' }}
        className="
          absolute -top-8 left-1/2 -translate-x-1/2
          w-11 h-11 rounded-xl
          bg-gradient-to-br from-accent/25 to-accent/10
          border border-accent/30
          flex items-center justify-center
          text-accent
          pointer-events-none
        "
      >
        {icon}
      </motion.div>

      {/* Card Content */}
      <motion.div
        animate={isExpanded ? { y: -4 } : { y: 0 }}
        className="
          w-full max-w-sm px-6 py-6 rounded-2xl
          bg-gradient-to-b from-white/[0.05] to-white/[0.01]
          border border-white/[0.08]
          hover:border-accent/30
          backdrop-blur-xl
          transition-all duration-300
          cursor-pointer
        "
      >
        {/* Header */}
        <div className="mb-3">
          <h3 className="text-xl font-semibold text-foreground mb-2 leading-tight">
            {title}
          </h3>
          <div className="flex items-center gap-2 text-xs text-foreground-muted mb-2">
            <Clock className="w-3.5 h-3.5" />
            <span>{duration}</span>
          </div>
        </div>

        {/* Description */}
        <p className="text-sm text-foreground-muted leading-relaxed">
          {description}
        </p>

        {/* Expandable Details on Hover */}
        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
              className="overflow-hidden"
            >
              <div className="pt-4 mt-4 border-t border-white/[0.06] space-y-2">
                {details.map((detail, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.05, duration: 0.2 }}
                    className="flex items-start gap-2 text-xs text-foreground-muted"
                  >
                    <div className="w-1 h-1 rounded-full bg-accent/60 mt-1.5 flex-shrink-0" />
                    <span>{detail}</span>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Connecting Line */}
      {!isLast && (
        <div className="absolute top-8 left-full w-12 h-px hidden md:block" style={{ 
          width: 'calc(100% - 2rem)',
          right: 'calc(-100% + 2rem)'
        }}>
          <motion.div
            initial={{ scaleX: 0 }}
            whileInView={{ scaleX: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: delay + 0.2 }}
            className="h-full bg-gradient-to-r from-accent/60 to-accent/30 origin-left"
          />
        </div>
      )}
    </motion.div>
  )
}

// ============================================================================
// MAIN DEPLOYMENT ROADMAP SECTION
// ============================================================================
export function DeploymentRoadmap() {
  const steps = [
    {
      step: 1,
      icon: <Plug className="w-6 h-6" />,
      title: "Plug",
      duration: "~30 seconds",
      description: "Connect to existing optical feeds (CCTV/IP/Webcam).",
      details: [
        "Support for 99% of camera models",
        "RTSP/ONVIF/HTTP streaming protocols",
        "Auto-detect resolution & frame rate",
        "Zero hardware installation required"
      ]
    },
    {
      step: 2,
      icon: <Zap className="w-6 h-6" />,
      title: "Calibrate",
      duration: "~30 seconds",
      description: "AI auto-maps 'Safe Zones' and 'Critical Zones' in seconds.",
      details: [
        "Computer vision analyzes room layout",
        "Identifies bed position & critical areas",
        "Learns baseline patient position",
        "Adjusts for lighting conditions"
      ]
    },
    {
      step: 3,
      icon: <Shield className="w-6 h-6" />,
      title: "Protect",
      duration: "Instant",
      description: "Live monitoring begins immediately. Alerts sent to Nursing Station.",
      details: [
        "Real-time video inference starts",
        "Multi-channel alert system active",
        "Nurse dashboard receives events",
        "Continuous learning mode enabled"
      ]
    },
  ]

  return (
    <section id="roadmap" className="relative py-24 md:py-32">
      {/* Top gradient line */}
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          className="text-center mb-20"
        >
          <span className="inline-block text-xs font-mono uppercase tracking-widest text-accent mb-4">
            Deployment Process
          </span>
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-semibold tracking-tight mb-4">
            <span className="bg-gradient-to-b from-white via-white/95 to-white/70 bg-clip-text text-transparent">
              Three Steps to Safety
            </span>
          </h2>
          <p className="text-lg text-foreground-muted max-w-2xl mx-auto">
            Operational in under 2 minutes. Hover over each step to explore details.
          </p>
        </motion.div>

        {/* Timeline Container */}
        <div className="relative">
          {/* Steps Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-12 md:gap-4">
            {steps.map((stepData, index) => (
              <div key={stepData.step} className="relative md:px-4">
                <InteractiveStepCard
                  {...stepData}
                  delay={index * 0.15}
                  isLast={index === steps.length - 1}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Simple CTA */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="text-center mt-20"
        >
          <a
            href="/dashboard"
            className="
              inline-flex items-center gap-2 px-8 py-3 rounded-xl
              bg-accent/10 border border-accent/40
              text-accent font-medium
              hover:bg-accent/20 hover:border-accent/60
              transition-all duration-300
            "
          >
            View Live Dashboard
            <ChevronRight className="w-4 h-4" />
          </a>
        </motion.div>
      </div>
    </section>
  )
}

export default DeploymentRoadmap
