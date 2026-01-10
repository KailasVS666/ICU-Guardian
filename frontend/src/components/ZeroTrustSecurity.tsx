/**
 * Zero-Trust Security Section - ICU Guardian (Redesigned)
 * 
 * Data Flow Architecture Visualization
 * Shows local processing pipeline: Capture → Process → Encrypt → Discard
 * Left: Text content with badges
 * Right: Animated data lifecycle diagram
 */

import { motion } from 'framer-motion'
import { Shield, Lock, CheckCircle2, Camera, Zap, Trash2, Server } from 'lucide-react'

// ============================================================================
// DATA FLOW ARCHITECTURE DIAGRAM
// ============================================================================
function DataFlowDiagram() {
  const dataVariants = {
    flow: {
      x: [0, 80, 160, 240, 280],
      opacity: [0, 1, 1, 1, 0],
      transition: {
        duration: 4,
        repeat: Infinity,
        ease: 'easeInOut',
        delay: 0
      }
    }
  }

  const dataVariants2 = {
    flow: {
      x: [0, 80, 160, 240, 280],
      opacity: [0, 1, 1, 1, 0],
      transition: {
        duration: 4,
        repeat: Infinity,
        ease: 'easeInOut',
        delay: 1.5
      }
    }
  }

  return (
    <div className="space-y-8">
      {/* Main Data Flow Diagram */}
      <div className="relative h-64 md:h-72 bg-gradient-to-br from-accent/5 to-transparent rounded-2xl border border-accent/20 p-6 overflow-hidden flex flex-col justify-center">
        {/* Stage Boxes with Animated Flow */}
        <div className="flex items-center justify-between gap-2 md:gap-4">
          {/* Stage 1: Capture */}
          <motion.div
            whileHover={{ scale: 1.05 }}
            className="flex flex-col items-center gap-2 flex-1"
          >
            <motion.div
              animate={{ 
                boxShadow: [
                  '0 0 0 0 rgba(94, 106, 210, 0.4)',
                  '0 0 0 12px rgba(94, 106, 210, 0)',
                ]
              }}
              transition={{ duration: 2, repeat: Infinity, delay: 0 }}
              className="w-14 h-14 md:w-16 md:h-16 rounded-2xl bg-gradient-to-br from-accent/40 to-accent/10 border-2 border-accent/50 flex items-center justify-center flex-shrink-0"
            >
              <Camera className="w-7 h-7 md:w-8 md:h-8 text-accent" />
            </motion.div>
            <div className="text-center">
              <p className="text-xs font-semibold text-accent">Capture</p>
              <p className="text-xs text-foreground-muted hidden md:inline">Video Input</p>
            </div>
          </motion.div>

          {/* Arrow 1 */}
          <motion.div
            animate={{ x: [0, 4, 0] }}
            transition={{ duration: 2, repeat: Infinity, delay: 0 }}
            className="hidden md:block text-accent/60 flex-shrink-0"
          >
            <Zap className="w-5 h-5 rotate-90" />
          </motion.div>

          {/* Stage 2: Processing */}
          <motion.div
            whileHover={{ scale: 1.05 }}
            className="flex flex-col items-center gap-2 flex-1"
          >
            <motion.div
              animate={{ 
                boxShadow: [
                  '0 0 0 0 rgba(94, 106, 210, 0.4)',
                  '0 0 0 12px rgba(94, 106, 210, 0)',
                ]
              }}
              transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}
              className="w-14 h-14 md:w-16 md:h-16 rounded-2xl bg-gradient-to-br from-accent/40 to-accent/10 border-2 border-accent/50 flex items-center justify-center flex-shrink-0"
            >
              <Server className="w-7 h-7 md:w-8 md:h-8 text-accent" />
            </motion.div>
            <div className="text-center">
              <p className="text-xs font-semibold text-accent">Process</p>
              <p className="text-xs text-foreground-muted hidden md:inline">Edge Device</p>
            </div>
          </motion.div>

          {/* Arrow 2 */}
          <motion.div
            animate={{ x: [0, 4, 0] }}
            transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}
            className="hidden md:block text-accent/60 flex-shrink-0"
          >
            <Zap className="w-5 h-5 rotate-90" />
          </motion.div>

          {/* Stage 3: Encryption */}
          <motion.div
            whileHover={{ scale: 1.05 }}
            className="flex flex-col items-center gap-2 flex-1"
          >
            <motion.div
              animate={{ 
                boxShadow: [
                  '0 0 0 0 rgba(94, 106, 210, 0.4)',
                  '0 0 0 12px rgba(94, 106, 210, 0)',
                ]
              }}
              transition={{ duration: 2, repeat: Infinity, delay: 1 }}
              className="w-14 h-14 md:w-16 md:h-16 rounded-2xl bg-gradient-to-br from-accent/40 to-accent/10 border-2 border-accent/50 flex items-center justify-center flex-shrink-0"
            >
              <Lock className="w-7 h-7 md:w-8 md:h-8 text-accent" />
            </motion.div>
            <div className="text-center">
              <p className="text-xs font-semibold text-accent">Secure</p>
              <p className="text-xs text-foreground-muted hidden md:inline">AES-256</p>
            </div>
          </motion.div>

          {/* Arrow 3 */}
          <motion.div
            animate={{ x: [0, 4, 0] }}
            transition={{ duration: 2, repeat: Infinity, delay: 1 }}
            className="hidden md:block text-accent/60 flex-shrink-0"
          >
            <Zap className="w-5 h-5 rotate-90" />
          </motion.div>

          {/* Stage 4: Deletion */}
          <motion.div
            whileHover={{ scale: 1.05 }}
            className="flex flex-col items-center gap-2 flex-1"
          >
            <motion.div
              animate={{ 
                boxShadow: [
                  '0 0 0 0 rgba(94, 106, 210, 0.4)',
                  '0 0 0 12px rgba(94, 106, 210, 0)',
                ]
              }}
              transition={{ duration: 2, repeat: Infinity, delay: 1.5 }}
              className="w-14 h-14 md:w-16 md:h-16 rounded-2xl bg-gradient-to-br from-accent/40 to-accent/10 border-2 border-accent/50 flex items-center justify-center flex-shrink-0"
            >
              <Trash2 className="w-7 h-7 md:w-8 md:h-8 text-accent" />
            </motion.div>
            <div className="text-center">
              <p className="text-xs font-semibold text-accent">Discard</p>
              <p className="text-xs text-foreground-muted hidden md:inline">Instantly</p>
            </div>
          </motion.div>
        </div>

        {/* Data Flow Particles */}
        <div className="mt-6 h-10 relative rounded-xl bg-gradient-to-r from-accent/10 via-transparent to-accent/10 border border-accent/20 overflow-hidden">
          {/* Flow Line */}
          <div className="absolute inset-0 flex items-center">
            <svg className="w-full h-full" viewBox="0 0 400 20" preserveAspectRatio="none">
              <line x1="0" y1="10" x2="400" y2="10" stroke="rgba(94,106,210,0.2)" strokeWidth="1" strokeDasharray="10,5" />
            </svg>
          </div>

          {/* Animated Data Particles */}
          <motion.div
            variants={dataVariants}
            animate="flow"
            className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-accent/80 shadow-lg shadow-accent/50"
          />
          <motion.div
            variants={dataVariants2}
            animate="flow"
            className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-accent/80 shadow-lg shadow-accent/50"
          />
        </div>

        {/* Info Text */}
        <p className="text-xs text-foreground-muted text-center mt-4">
          Data flows through edge device, processed locally, encrypted for safety, then instantly deleted. <span className="text-accent font-semibold">No cloud, no storage, no permanent records.</span>
        </p>
      </div>

      {/* Performance Metrics - Visual Bar Representation */}
      <div className="space-y-6">
        {/* Latency Metric */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-foreground">Processing Latency</span>
            <span className="text-2xl font-bold font-mono bg-gradient-to-r from-accent to-accent/70 bg-clip-text text-transparent">
              15ms
            </span>
          </div>
          <div className="relative h-2 rounded-full bg-white/[0.05] overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              whileInView={{ width: '15%' }}
              viewport={{ once: true }}
              transition={{ duration: 1.5, ease: [0.16, 1, 0.3, 1], delay: 0.2 }}
              className="absolute h-full rounded-full bg-gradient-to-r from-accent via-accent-bright to-accent"
              style={{ boxShadow: '0 0 20px rgba(94, 106, 210, 0.5)' }}
            />
          </div>
          <p className="text-xs text-foreground-muted">Ultra-fast edge processing</p>
        </div>

        {/* Cloud Dependency Metric */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-foreground">Cloud Dependency</span>
            <span className="text-2xl font-bold font-mono bg-gradient-to-r from-accent to-accent/70 bg-clip-text text-transparent">
              0%
            </span>
          </div>
          <div className="relative h-2 rounded-full bg-white/[0.05] overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              whileInView={{ width: '0%' }}
              viewport={{ once: true }}
              transition={{ duration: 1.5, ease: [0.16, 1, 0.3, 1], delay: 0.4 }}
              className="absolute h-full rounded-full bg-gradient-to-r from-accent via-accent-bright to-accent"
            />
            {/* Empty state indicator */}
            <div className="absolute right-2 top-1/2 -translate-y-1/2">
              <div className="w-1.5 h-1.5 rounded-full bg-accent/60 animate-pulse" />
            </div>
          </div>
          <p className="text-xs text-foreground-muted">100% local processing, zero cloud reliance</p>
        </div>

        {/* Local Control Metric */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-foreground">Local Control</span>
            <span className="text-2xl font-bold font-mono bg-gradient-to-r from-accent to-accent/70 bg-clip-text text-transparent">
              100%
            </span>
          </div>
          <div className="relative h-2 rounded-full bg-white/[0.05] overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              whileInView={{ width: '100%' }}
              viewport={{ once: true }}
              transition={{ duration: 1.5, ease: [0.16, 1, 0.3, 1], delay: 0.6 }}
              className="absolute h-full rounded-full bg-gradient-to-r from-accent via-accent-bright to-accent"
              style={{ boxShadow: '0 0 20px rgba(94, 106, 210, 0.5)' }}
            />
            <motion.div
              animate={{ x: [-10, 10, -10] }}
              transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
              className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
              style={{ width: '30px' }}
            />
          </div>
          <p className="text-xs text-foreground-muted">Complete data sovereignty at the edge</p>
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// SECURITY BADGE COMPONENT
// ============================================================================
interface SecurityBadgeProps {
  icon: React.ReactNode
  label: string
  description: string
  delay: number
}

function SecurityBadge({ icon, label, description, delay }: SecurityBadgeProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay }}
      className="
        group flex items-center gap-3 px-4 py-3 rounded-xl
        bg-gradient-to-r from-white/[0.02] to-white/[0.01]
        border border-accent/30 hover:border-accent/60
        transition-all duration-300
        hover:bg-accent/10
      "
    >
      <div className="flex-shrink-0 text-accent group-hover:text-accent-bright transition-colors">
        {icon}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-foreground">{label}</p>
        <p className="text-xs text-foreground-muted">{description}</p>
      </div>
      <motion.div
        animate={{ scale: [1, 1.2, 1] }}
        transition={{ duration: 2, repeat: Infinity, delay }}
        className="flex-shrink-0"
      >
        <div className="w-1.5 h-1.5 rounded-full bg-accent/60" />
      </motion.div>
    </motion.div>
  )
}

// ============================================================================
// MAIN ZERO-TRUST SECURITY SECTION
// ============================================================================
export function ZeroTrustSecurity() {
  return (
    <section id="security" className="relative py-24 md:py-32">
      {/* Top gradient line */}
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-16 items-center">
          {/* Left: Text Content */}
          <motion.div
            initial={{ opacity: 0, x: -32 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
            className="space-y-8"
          >
            {/* Label */}
            <motion.div
              initial={{ opacity: 0, y: -16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4 }}
              className="inline-flex items-center gap-2"
            >
              <Lock className="w-4 h-4 text-accent" />
              <span className="text-xs font-mono uppercase tracking-widest text-accent">
                Zero-Trust Architecture
              </span>
            </motion.div>

            {/* Headline */}
            <h2 className="text-3xl md:text-4xl lg:text-5xl font-semibold tracking-tight">
              <span className="bg-gradient-to-b from-white via-white/95 to-white/70 bg-clip-text text-transparent">
                Your Data Never Leaves the Room.
              </span>
            </h2>

            {/* Subheadline */}
            <p className="text-lg text-foreground-muted leading-relaxed">
              Built on a <span className="text-foreground font-semibold">"Process & Discard"</span> architecture. Video feeds are analyzed locally on Edge devices and instantly deleted. No cloud uploads. No permanent recording.
            </p>

            {/* Security Badges */}
            <div className="space-y-3 pt-4">
              <SecurityBadge
                icon={<Shield className="w-4 h-4" />}
                label="HIPAA Compliant"
                description="Healthcare data protection standards"
                delay={0}
              />
              <SecurityBadge
                icon={<Lock className="w-4 h-4" />}
                label="GDPR Ready"
                description="European privacy regulations"
                delay={0.1}
              />
              <SecurityBadge
                icon={<CheckCircle2 className="w-4 h-4" />}
                label="AES-256 Encryption"
                description="Military-grade data security"
                delay={0.2}
              />
            </div>

            {/* Additional trust statement */}
            <div className="pt-4 border-t border-white/[0.06]">
              <p className="text-sm text-foreground-muted">
                ✓ All processing happens at the edge <br />
                ✓ Zero data collection policy <br />
                ✓ Real-time audit logging <br />
                ✓ Compliant with global privacy laws
              </p>
            </div>
          </motion.div>

          {/* Right: Visual - Data Flow Diagram */}
          <motion.div
            initial={{ opacity: 0, x: 32 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1], delay: 0.1 }}
          >
            <DataFlowDiagram />
          </motion.div>
        </div>
      </div>
    </section>
  )
}

export default ZeroTrustSecurity
