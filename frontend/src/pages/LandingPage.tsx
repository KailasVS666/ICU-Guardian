/**
 * ICU Guardian Landing Page
 * 
 * Design: Linear Design System with mouse-tracking spotlight effects
 * Premium, cinematic dark interface with ambient lighting
 */

import { useState, useEffect, useRef, MouseEvent } from 'react'
import { Link } from 'react-router-dom'
import { motion, useScroll, useTransform } from 'framer-motion'
import { 
  Shield, 
  Activity, 
  Eye, 
  Lock, 
  Menu, 
  X,
  ChevronRight,
  Terminal,
  AlertTriangle,
  Linkedin,
  Cctv,
  Webcam,
  Video,
  FileText,
  Clock,
  Download
} from 'lucide-react'
import { AmbientBackground } from '../components/ui'
import { SecureCoreVisualization } from '../components/SecureCoreVisualization'
import { CapabilitiesBentoGrid } from '../components/CapabilitiesBentoGrid'
import { ZeroTrustSecurity } from '../components/ZeroTrustSecurity'
import { DeploymentRoadmap } from '../components/DeploymentRoadmap'
import { ImpactStatistics } from '../components/ImpactStatistics'
import { ScrollToTop } from '../components/ScrollToTop'

// ============================================================================
// DESIGN TOKENS (Matching Dashboard - Linear Design System)
// ============================================================================
const COLORS = {
  // Primary accent (indigo - matching dashboard)
  accent: '#5E6AD2',
  accentBright: '#6872D9',
  accentGlow: 'rgba(94, 106, 210, 0.3)',
  // Alert colors
  clinicalRed: '#FF4B4B',
  // Base colors
  backgroundDeep: '#020203',
  backgroundBase: '#050506',
  foreground: '#EDEDEF',
  foregroundMuted: '#8A8F98',
}

// ============================================================================
// ANIMATION VARIANTS
// ============================================================================
const fadeInUp = {
  initial: { opacity: 0, y: 24 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6, ease: [0.16, 1, 0.3, 1] }
}

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.08,
      delayChildren: 0.1
    }
  }
}

// ============================================================================
// SPOTLIGHT CARD - Mouse tracking glow effect (matching dashboard)
// ============================================================================
interface SpotlightCardProps {
  children: React.ReactNode
  className?: string
}

function SpotlightCard({ children, className = '' }: SpotlightCardProps) {
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
      className={`spotlight-card glass-card ${className}`}
    >
      {children}
    </div>
  )
}

// ============================================================================
// NAVBAR COMPONENT
// ============================================================================
function Navbar() {
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  // Smooth scroll handler
  const handleSmoothScroll = (e: React.MouseEvent<HTMLAnchorElement>, href: string) => {
    if (href.startsWith('#')) {
      e.preventDefault()
      const targetId = href.substring(1)
      const targetElement = document.getElementById(targetId)
      if (targetElement) {
        targetElement.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        })
      }
      setIsMenuOpen(false)
    }
  }

  const navLinks = [
    { label: 'Features', href: '#features' },
    { label: 'Team', href: '#team' },
    { label: 'Tech', href: '#tech' },
  ]

  return (
    <motion.nav
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled 
          ? 'bg-background-base/80 backdrop-blur-xl border-b border-white/[0.06]' 
          : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 lg:h-20">
          {/* Logo */}
          <a href="#" className="flex items-center gap-3 group">
            <div className="relative">
              <Shield className="w-7 h-7 text-accent" />
              {/* Pulsing glow */}
              <div className="absolute inset-0 animate-pulse">
                <Shield className="w-7 h-7 text-accent blur-md opacity-50" />
              </div>
            </div>
            <span className="text-xl font-bold tracking-tighter text-foreground">
              ICU Guardian
            </span>
          </a>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-8">
            {navLinks.map((link) => (
              <a
                key={link.label}
                href={link.href}
                target={link.external ? '_blank' : undefined}
                rel={link.external ? 'noopener noreferrer' : undefined}
                onClick={(e) => handleSmoothScroll(e, link.href)}
                className="text-sm text-foreground-muted hover:text-foreground transition-colors duration-200 cursor-pointer"
              >
                {link.label}
              </a>
            ))}
            
            {/* Launch Dashboard CTA */}
            <Link
              to="/dashboard"
              className="
                relative group px-5 py-2.5 rounded-lg
                bg-transparent border border-accent/50
                text-accent font-medium text-sm
                transition-all duration-300
                hover:border-accent hover:bg-accent/10
                hover:shadow-accent-glow
              "
            >
              <span className="relative z-10">Launch Dashboard</span>
            </Link>
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            className="md:hidden p-2 text-foreground-muted hover:text-foreground transition-colors"
          >
            {isMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>

        {/* Mobile Menu */}
        <motion.div
          initial={false}
          animate={{ height: isMenuOpen ? 'auto' : 0, opacity: isMenuOpen ? 1 : 0 }}
          transition={{ duration: 0.2 }}
          className="md:hidden overflow-hidden"
        >
          <div className="py-4 space-y-4 border-t border-white/[0.06]">
            {navLinks.map((link) => (
              <a
                key={link.label}
                href={link.href}
                target={link.external ? '_blank' : undefined}
                className="block text-foreground-muted hover:text-foreground transition-colors cursor-pointer"
                onClick={(e) => handleSmoothScroll(e, link.href)}
              >
                {link.label}
              </a>
            ))}
            <Link
              to="/dashboard"
              className="
                block w-full text-center px-5 py-3 rounded-lg
                bg-accent/10 border border-accent/50
                text-accent font-medium
              "
              onClick={() => setIsMenuOpen(false)}
            >
              Launch Dashboard
            </Link>
          </div>
        </motion.div>
      </div>
    </motion.nav>
  )
}

// ============================================================================
// HERO SECTION - "The Sentinel"
// ============================================================================
function HeroSection() {
  const containerRef = useRef<HTMLDivElement>(null)
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end start"]
  })
  
  const opacity = useTransform(scrollYProgress, [0, 0.5], [1, 0])
  const scale = useTransform(scrollYProgress, [0, 0.5], [1, 0.95])
  const y = useTransform(scrollYProgress, [0, 0.5], [0, 100])

  return (
    <section 
      ref={containerRef}
      className="relative min-h-screen flex items-center justify-center overflow-hidden"
    >
      {/* Scanning Grid Animation */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {/* Grid pattern - using accent color */}
        <div 
          className="absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage: `
              linear-gradient(rgba(94, 106, 210, 0.3) 1px, transparent 1px),
              linear-gradient(90deg, rgba(94, 106, 210, 0.3) 1px, transparent 1px)
            `,
            backgroundSize: '60px 60px'
          }}
        />
        
        {/* Scanning laser line */}
        <motion.div
          animate={{ y: ['0%', '100%', '0%'] }}
          transition={{ duration: 8, ease: 'linear', repeat: Infinity }}
          className="absolute left-0 right-0 h-[2px] shadow-[0_0_30px_10px_rgba(94,106,210,0.3)]"
          style={{ background: `linear-gradient(90deg, transparent, ${COLORS.accent}, transparent)` }}
        />

        {/* Target brackets */}
        <TargetBrackets />
      </div>

      {/* Hero Content */}
      <motion.div 
        style={{ opacity, scale, y }}
        className="relative z-10 max-w-5xl mx-auto px-4 text-center"
      >
        <motion.div
          initial="initial"
          animate="animate"
          variants={staggerContainer}
          className="space-y-8"
        >
          {/* Label */}
          <motion.div 
            variants={fadeInUp}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/[0.03] border border-white/[0.06]"
          >
            <div className="w-2 h-2 rounded-full bg-accent animate-pulse" />
            <span className="text-xs font-mono uppercase tracking-widest text-foreground-muted">
              AI-Powered ICU Monitoring
            </span>
          </motion.div>

          {/* Headline */}
          <motion.h1 
            variants={fadeInUp}
            className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-semibold tracking-tight leading-[1.1]"
          >
            <span className="bg-gradient-to-b from-white via-white/95 to-white/70 bg-clip-text text-transparent">
              The Second Pair of Eyes
            </span>
            <br />
            <span className="bg-gradient-to-b from-white/90 via-white/70 to-white/50 bg-clip-text text-transparent">
              Every ICU Needs.
            </span>
          </motion.h1>

          {/* Sub-headline */}
          <motion.p 
            variants={fadeInUp}
            className="max-w-2xl mx-auto text-lg md:text-xl text-foreground-muted leading-relaxed"
          >
            Predict delirium <span className="text-accent font-medium">2 hours early</span>. 
            Detect self-extubation <span className="text-accent font-medium">instantly</span>. 
            Powered by privacy-first Computer Vision.
          </motion.p>

          {/* CTA Button */}
          <motion.div variants={fadeInUp}>
            <Link
              to="/dashboard"
              className="
                inline-flex items-center gap-3 px-8 py-4 rounded-xl
                bg-accent text-white font-semibold text-lg
                transition-all duration-300
                hover:scale-[1.02] active:scale-[0.98]
                hover:bg-accent-bright
                shadow-accent-glow
              "
            >
              Start Monitoring
              <ChevronRight className="w-5 h-5" />
            </Link>
          </motion.div>
        </motion.div>
      </motion.div>

      {/* Scroll indicator */}
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5 }}
        className="absolute bottom-8 left-1/2 -translate-x-1/2"
      >
        <motion.div
          animate={{ y: [0, 8, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="w-6 h-10 rounded-full border-2 border-white/20 flex items-start justify-center p-2"
        >
          <div className="w-1.5 h-2.5 rounded-full bg-white/40" />
        </motion.div>
      </motion.div>
    </section>
  )
}

// Floating target brackets for hero
function TargetBrackets() {
  const positions = [
    { top: '25%', left: '15%', delay: 0 },
    { top: '35%', right: '20%', delay: 0.5 },
    { top: '60%', left: '25%', delay: 1 },
    { top: '70%', right: '30%', delay: 1.5 },
  ]

  return (
    <>
      {positions.map((pos, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ 
            opacity: [0, 0.6, 0.6, 0],
            scale: [0.8, 1, 1, 0.9]
          }}
          transition={{
            duration: 4,
            delay: pos.delay,
            repeat: Infinity,
            repeatDelay: 2
          }}
          className="absolute w-12 h-12 text-accent/60"
          style={{ top: pos.top, left: pos.left, right: pos.right }}
        >
          {/* Target bracket corners */}
          <div className="absolute top-0 left-0 w-3 h-3 border-t-2 border-l-2 border-accent" />
          <div className="absolute top-0 right-0 w-3 h-3 border-t-2 border-r-2 border-accent" />
          <div className="absolute bottom-0 left-0 w-3 h-3 border-b-2 border-l-2 border-accent" />
          <div className="absolute bottom-0 right-0 w-3 h-3 border-b-2 border-r-2 border-accent" />
        </motion.div>
      ))}
    </>
  )
}

// ============================================================================
// FEATURES SECTION - Bento Grid
// ============================================================================
// Kinetic Vector Analysis Card with Waveform Animation
function KineticVectorCard() {
  const cardRef = useRef<HTMLDivElement>(null)
  const [isHovered, setIsHovered] = useState(false)

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return
    const rect = cardRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    cardRef.current.style.setProperty('--mouse-x', `${x}px`)
    cardRef.current.style.setProperty('--mouse-y', `${y}px`)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-50px" }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      className="lg:col-span-2 lg:row-span-2"
    >
      <div
        ref={cardRef}
        onMouseMove={handleMouseMove}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        className={`
          spotlight-card glass-card
          group relative p-8 h-full overflow-hidden
          hover:-translate-y-1 transition-all duration-300
          ${isHovered ? 'border-[#FF4B4B]/50 shadow-[0_0_40px_rgba(255,75,75,0.15)]' : ''}
        `}
      >
        {/* Animated Waveform Background */}
        <div className="absolute inset-0 overflow-hidden opacity-20 group-hover:opacity-40 transition-opacity duration-500">
          <svg
            className="absolute bottom-0 left-0 w-full h-32"
            viewBox="0 0 400 100"
            preserveAspectRatio="none"
          >
            {/* Calm breathing wave */}
            <motion.path
              d="M0,50 Q25,45 50,50 T100,50 T150,50 T200,50 T250,50 T300,50 T350,50 T400,50"
              fill="none"
              stroke={isHovered ? '#FF4B4B' : '#5E6AD2'}
              strokeWidth="2"
              animate={{
                d: isHovered 
                  ? [
                      "M0,50 Q25,45 50,50 T100,50 T150,50 T200,50 T250,50 T300,50 T350,50 T400,50",
                      "M0,50 Q25,30 50,50 T100,20 T150,80 T200,10 T250,90 T300,15 T350,50 T400,50",
                      "M0,50 Q25,45 50,50 T100,50 T150,50 T200,50 T250,50 T300,50 T350,50 T400,50"
                    ]
                  : [
                      "M0,50 Q25,45 50,50 T100,50 T150,50 T200,50 T250,50 T300,50 T350,50 T400,50",
                      "M0,50 Q25,40 50,50 T100,45 T150,55 T200,45 T250,55 T300,45 T350,50 T400,50",
                      "M0,50 Q25,45 50,50 T100,50 T150,50 T200,50 T250,50 T300,50 T350,50 T400,50"
                    ]
              }}
              transition={{
                duration: isHovered ? 0.5 : 3,
                repeat: Infinity,
                ease: isHovered ? "easeOut" : "easeInOut"
              }}
            />
            {/* Secondary wave */}
            <motion.path
              d="M0,60 Q25,55 50,60 T100,60 T150,60 T200,60 T250,60 T300,60 T350,60 T400,60"
              fill="none"
              stroke={isHovered ? '#FF4B4B' : '#5E6AD2'}
              strokeWidth="1"
              opacity="0.5"
              animate={{
                d: isHovered
                  ? [
                      "M0,60 Q25,55 50,60 T100,60 T150,60 T200,60 T250,60 T300,60 T350,60 T400,60",
                      "M0,60 Q25,40 50,60 T100,30 T150,70 T200,25 T250,75 T300,35 T350,60 T400,60"
                    ]
                  : [
                      "M0,60 Q25,55 50,60 T100,60 T150,60 T200,60 T250,60 T300,60 T350,60 T400,60",
                      "M0,60 Q25,50 50,60 T100,55 T150,65 T200,55 T250,65 T300,55 T350,60 T400,60"
                    ]
              }}
              transition={{
                duration: isHovered ? 0.4 : 2.5,
                repeat: Infinity,
                ease: "easeInOut",
                delay: 0.2
              }}
            />
          </svg>
        </div>

        {/* Alert flash overlay on hover */}
        <motion.div
          className="absolute inset-0 rounded-2xl pointer-events-none"
          animate={{
            backgroundColor: isHovered 
              ? ['rgba(255,75,75,0)', 'rgba(255,75,75,0.08)', 'rgba(255,75,75,0)']
              : 'rgba(255,75,75,0)'
          }}
          transition={{
            duration: 0.8,
            repeat: isHovered ? Infinity : 0,
            ease: "easeInOut"
          }}
        />

        {/* Icon */}
        <div className={`
          inline-flex items-center justify-center w-12 h-12 rounded-xl mb-6
          transition-all duration-300
          ${isHovered 
            ? 'bg-[#FF4B4B]/15 border border-[#FF4B4B]/30' 
            : 'bg-accent/10 border border-accent/20 group-hover:bg-accent/15 group-hover:border-accent/30'
          }
        `}>
          <Activity className={`w-6 h-6 transition-colors duration-300 ${isHovered ? 'text-[#FF4B4B]' : 'text-accent'}`} />
        </div>

        {/* Content */}
        <h3 className="font-semibold tracking-tight text-foreground mb-3 text-2xl lg:text-3xl">
          Kinetic Vector Analysis
        </h3>
        <p className="text-foreground-muted leading-relaxed text-lg max-w-xl">
          Differentiates benign rhythmic respiratory motion from acute agitation events. 
          Our engine analyzes velocity-acceleration vectors in real-time to filter out 99% of false motion alarms.
        </p>

        {/* Status indicator */}
        <div className={`
          mt-6 inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium
          transition-all duration-300
          ${isHovered 
            ? 'bg-[#FF4B4B]/10 text-[#FF4B4B] border border-[#FF4B4B]/30' 
            : 'bg-accent/10 text-accent border border-accent/20'
          }
        `}>
          <span className={`w-2 h-2 rounded-full ${isHovered ? 'bg-[#FF4B4B] animate-pulse' : 'bg-accent'}`} />
          {isHovered ? 'Agitation Detected' : 'Monitoring Active'}
        </div>

        {/* Accent glow */}
        <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none">
          <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[400px] h-[400px] rounded-full blur-[100px] transition-colors duration-300 ${isHovered ? 'bg-[#FF4B4B]/10' : 'bg-accent/5'}`} />
        </div>
      </div>
    </motion.div>
  )
}

function FeaturesSection() {
  const features = [
    {
      icon: Eye,
      title: 'Omni-Vision',
      description: 'Skeletal tracking detects hands nearing breathing tubes with sub-second response time.',
      size: 'normal',
    },
    {
      icon: Lock,
      title: 'Privacy Shield',
      description: 'Smart blurring protects patient identity until a threat is confirmed. HIPAA-ready by design.',
      size: 'normal',
    },
  ]

  return (
    <section id="features" className="relative py-24 md:py-32">
      {/* Section divider */}
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          className="text-center mb-16"
        >
          <span className="text-xs font-mono uppercase tracking-widest text-accent mb-4 block">
            The Intelligence
          </span>
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-semibold tracking-tight bg-gradient-to-b from-white to-white/70 bg-clip-text text-transparent">
            Three Pillars of Protection
          </h2>
        </motion.div>

        {/* Bento Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Special Kinetic Vector Card */}
          <KineticVectorCard />
          
          {/* Other feature cards */}
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, delay: (index + 1) * 0.1, ease: [0.16, 1, 0.3, 1] }}
            >
              <SpotlightCard 
                className={`
                  group relative p-8 h-full
                  hover:-translate-y-1
                `}
              >
                {/* Icon */}
                <div className="
                  inline-flex items-center justify-center w-12 h-12 rounded-xl mb-6
                  bg-accent/10 border border-accent/20
                  group-hover:bg-accent/15 group-hover:border-accent/30
                  transition-colors duration-300
                ">
                  <feature.icon className="w-6 h-6 text-accent" />
                </div>

                {/* Content */}
                <h3 className="font-semibold tracking-tight text-foreground mb-3 text-xl">
                  {feature.title}
                </h3>
                <p className="text-foreground-muted leading-relaxed text-base">
                  {feature.description}
                </p>
              </SpotlightCard>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}

// ============================================================================
// DR. AI SECTION - Chat Interface
// ============================================================================
function DrAISection() {
  const [displayedText, setDisplayedText] = useState('')
  const fullText = "⚠️ Analysis Complete: SpO2 dropping (92%). Agitation detected. Recommend immediate restraint check."
  const hasTyped = useRef(false)

  useEffect(() => {
    if (hasTyped.current) return
    
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !hasTyped.current) {
          hasTyped.current = true
          let i = 0
          const interval = setInterval(() => {
            if (i < fullText.length) {
              setDisplayedText(fullText.slice(0, i + 1))
              i++
            } else {
              clearInterval(interval)
            }
          }, 40)
        }
      },
      { threshold: 0.5 }
    )

    const section = document.getElementById('dr-ai-section')
    if (section) observer.observe(section)

    return () => observer.disconnect()
  }, [])

  return (
    <section id="dr-ai-section" className="relative py-24 md:py-32">
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
          {/* Left - Explanation */}
          <motion.div
            initial={{ opacity: 0, x: -24 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          >
            <span className="text-xs font-mono uppercase tracking-widest text-accent mb-4 block">
              AI Integration
            </span>
            <h2 className="text-3xl md:text-4xl font-semibold tracking-tight mb-6 bg-gradient-to-b from-white to-white/70 bg-clip-text text-transparent">
              Dr. AI: Holistic Risk Assessment
            </h2>
            <p className="text-lg text-foreground-muted leading-relaxed mb-6">
              Our system fuses <span className="text-accent">real-time vitals data</span> with{' '}
              <span className="text-accent">computer vision analysis</span> to generate 
              a holistic risk score that predicts critical events before they happen.
            </p>
            <ul className="space-y-4">
              {[
                'SpO2, heart rate, and respiratory patterns',
                'Body posture and movement analysis',
                'Facial distress detection',
                'Hand proximity to critical areas'
              ].map((item, i) => (
                <li key={i} className="flex items-center gap-3 text-foreground-muted">
                  <div className="w-1.5 h-1.5 rounded-full bg-accent" />
                  {item}
                </li>
              ))}
            </ul>
          </motion.div>

          {/* Right - Chat Interface */}
          <motion.div
            initial={{ opacity: 0, x: 24 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="relative"
          >
            <SpotlightCard className="p-6">
              {/* Terminal header */}
              <div className="flex items-center gap-3 mb-4 pb-4 border-b border-white/[0.06]">
                <div className="flex gap-2">
                  <div className="w-3 h-3 rounded-full bg-status-error/80" />
                  <div className="w-3 h-3 rounded-full bg-status-warning/80" />
                  <div className="w-3 h-3 rounded-full bg-status-success/80" />
                </div>
                <span className="text-xs font-mono text-foreground-muted">dr-ai-analysis.log</span>
              </div>

              {/* Terminal content */}
              <div className="font-mono text-sm">
                <div className="flex items-center gap-2 text-foreground-muted mb-2">
                  <Terminal className="w-4 h-4" />
                  <span>Running analysis...</span>
                </div>
                
                <div className="
                  p-4 rounded-lg
                  bg-background-elevated/50
                  border border-status-error/20
                  min-h-[80px]
                ">
                  <div className="flex items-start gap-2">
                    <AlertTriangle className="w-4 h-4 text-status-error mt-0.5 flex-shrink-0" />
                    <span className="text-foreground leading-relaxed">
                      {displayedText}
                      <span className="inline-block w-2 h-4 bg-accent ml-1 animate-pulse" />
                    </span>
                  </div>
                </div>

                <div className="mt-4 pt-4 border-t border-white/[0.06] text-xs text-foreground-muted">
                  <span className="text-accent">Risk Level:</span> HIGH · 
                  <span className="ml-2 text-accent">Confidence:</span> 94.2%
                </div>
              </div>
            </SpotlightCard>

            {/* Decorative glow */}
            <div className="absolute -inset-4 rounded-3xl bg-status-error/5 blur-2xl -z-10" />
          </motion.div>
        </div>
      </div>
    </section>
  )
}

// ============================================================================
// ALARM FATIGUE SECTION - "The Silence of Safety"
// ============================================================================
function AlarmFatigueSection() {
  return (
    <section className="relative py-24 md:py-32">
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
          {/* Left - Noise Filtering Visualization */}
          <motion.div
            initial={{ opacity: 0, x: -24 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
            className="relative"
          >
            <SpotlightCard className="p-8 overflow-hidden">
              <div className="relative h-[200px] flex items-center">
                {/* Old Way - Chaotic Waveform (Left Side) */}
                <div className="flex-1 relative">
                  <span className="absolute -top-6 left-0 text-xs font-mono text-foreground-muted/60">
                    Traditional Monitors
                  </span>
                  <svg viewBox="0 0 120 80" className="w-full h-20" preserveAspectRatio="none">
                    <motion.path
                      d="M0,40 L5,35 L10,55 L15,25 L20,60 L25,20 L30,50 L35,30 L40,65 L45,15 L50,55 L55,28 L60,58 L65,22 L70,52 L75,32 L80,48 L85,38 L90,55 L95,25 L100,45 L105,35 L110,50 L115,30 L120,40"
                      fill="none"
                      stroke="#6B7280"
                      strokeWidth="2"
                      strokeLinecap="round"
                      initial={{ pathLength: 0 }}
                      whileInView={{ pathLength: 1 }}
                      viewport={{ once: true }}
                      transition={{ duration: 1.5, ease: "easeOut" }}
                    />
                    {/* Animated noise jitter */}
                    <motion.g
                      animate={{ 
                        scaleY: [1, 1.3, 0.8, 1.2, 1],
                        translateY: [0, -2, 3, -1, 0]
                      }}
                      transition={{ duration: 0.5, repeat: Infinity, repeatType: "reverse" }}
                      style={{ transformOrigin: 'center' }}
                    >
                      <path
                        d="M0,40 L5,35 L10,55 L15,25 L20,60 L25,20 L30,50 L35,30 L40,65 L45,15 L50,55 L55,28 L60,58"
                        fill="none"
                        stroke="#6B7280"
                        strokeWidth="1"
                        strokeLinecap="round"
                        opacity="0.3"
                      />
                    </motion.g>
                  </svg>
                  <div className="flex items-center gap-1 mt-2">
                    {[...Array(8)].map((_, i) => (
                      <motion.div
                        key={i}
                        animate={{ opacity: [0.3, 1, 0.3] }}
                        transition={{ duration: 0.3, delay: i * 0.1, repeat: Infinity }}
                        className="w-2 h-2 rounded-full bg-status-warning"
                      />
                    ))}
                    <span className="text-xs text-status-warning ml-2">350+ alarms/day</span>
                  </div>
                </div>

                {/* Middle - Scanning Filter Line */}
                <div className="relative w-16 h-full flex items-center justify-center mx-4">
                  <motion.div
                    className="absolute w-[2px] h-full rounded-full"
                    style={{ 
                      background: `linear-gradient(180deg, transparent 0%, ${COLORS.accent} 50%, transparent 100%)`,
                      boxShadow: `0 0 20px 5px ${COLORS.accentGlow}`
                    }}
                    animate={{ 
                      opacity: [0.5, 1, 0.5],
                      scaleY: [0.9, 1, 0.9]
                    }}
                    transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                  />
                  <motion.div
                    className="absolute w-8 h-8 rounded-full border-2 border-accent"
                    animate={{ scale: [1, 1.2, 1], opacity: [0.8, 0.4, 0.8] }}
                    transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                  />
                  <Shield className="w-5 h-5 text-accent relative z-10" />
                </div>

                {/* Right Side - Calm Line with Single Spike */}
                <div className="flex-1 relative">
                  <span className="absolute -top-6 right-0 text-xs font-mono text-accent">
                    ICU Guardian
                  </span>
                  <svg viewBox="0 0 120 80" className="w-full h-20" preserveAspectRatio="none">
                    {/* Flat calm line */}
                    <motion.path
                      d="M0,40 L30,40 L35,40 L40,40 L45,40 L50,40 L55,40 L60,40 L65,40 L70,40"
                      fill="none"
                      stroke={COLORS.accent}
                      strokeWidth="2"
                      strokeLinecap="round"
                      initial={{ pathLength: 0 }}
                      whileInView={{ pathLength: 1 }}
                      viewport={{ once: true }}
                      transition={{ duration: 1, delay: 0.5, ease: "easeOut" }}
                    />
                    {/* Single verified threat spike */}
                    <motion.path
                      d="M70,40 L75,40 L80,10 L85,70 L90,40 L120,40"
                      fill="none"
                      stroke={COLORS.clinicalRed}
                      strokeWidth="2.5"
                      strokeLinecap="round"
                      initial={{ pathLength: 0, opacity: 0 }}
                      whileInView={{ pathLength: 1, opacity: 1 }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.8, delay: 1.5, ease: "easeOut" }}
                    />
                  </svg>
                  <div className="flex items-center gap-2 mt-2 justify-end">
                    <motion.div
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 1.5, repeat: Infinity }}
                      className="w-2 h-2 rounded-full bg-status-error"
                    />
                    <span className="text-xs text-status-error">Verified Threat</span>
                  </div>
                </div>
              </div>

              {/* Stat Badge */}
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: 0.8 }}
                className="mt-8 flex justify-center"
              >
                <div className="
                  inline-flex items-center gap-2 px-5 py-2.5 rounded-full
                  bg-accent/10 border border-accent/30
                  shadow-[0_0_20px_rgba(94,106,210,0.2)]
                ">
                  <span className="text-2xl font-bold text-accent">94%</span>
                  <span className="text-sm text-foreground-muted">Reduction in False Alarms</span>
                </div>
              </motion.div>
            </SpotlightCard>
          </motion.div>

          {/* Right - Text Content */}
          <motion.div
            initial={{ opacity: 0, x: 24 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
          >
            <span className="text-xs font-mono uppercase tracking-widest text-accent mb-4 block">
              Combatting Alarm Fatigue
            </span>
            <h2 className="text-3xl md:text-4xl font-semibold tracking-tight mb-6 bg-gradient-to-b from-white to-white/70 bg-clip-text text-transparent">
              We curbed the boy who cried wolf.
            </h2>
            <p className="text-lg text-foreground-muted leading-relaxed mb-6">
              Standard monitors trigger <span className="text-accent font-medium">350+ false alarms</span> per bed, 
              per day. ICU Guardian filters out pillow adjustments, routine nursing care, and family 
              visits—alerting you <span className="text-foreground font-medium">only when it matters</span>.
            </p>
            <ul className="space-y-3">
              {[
                'Intelligent motion classification',
                'Context-aware activity recognition',
                'Nurse workflow integration',
                'Customizable sensitivity thresholds'
              ].map((item, i) => (
                <li key={i} className="flex items-center gap-3 text-foreground-muted">
                  <div className="w-1.5 h-1.5 rounded-full bg-accent" />
                  {item}
                </li>
              ))}
            </ul>
          </motion.div>
        </div>
      </div>
    </section>
  )
}

// ============================================================================
// INFRASTRUCTURE SECTION - "Infrastructure Independent"
// ============================================================================
function InfrastructureSection() {
  const cameraIcons = [
    { Icon: Cctv, label: 'CCTV' },
    { Icon: Webcam, label: 'Webcam' },
    { Icon: Video, label: 'IP Camera' },
  ]

  return (
    <section className="relative py-24 md:py-32">
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          className="text-center mb-16"
        >
          <span className="text-xs font-mono uppercase tracking-widest text-accent mb-4 block">
            Hardware Agnostic
          </span>
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-semibold tracking-tight mb-6 bg-gradient-to-b from-white to-white/70 bg-clip-text text-transparent">
            Works with the cameras you already own.
          </h2>
          <p className="text-lg text-foreground-muted max-w-2xl mx-auto">
            No expensive LIDAR. No wearable sensors. Deploy ICU Guardian on your existing optical 
            feed in under <span className="text-accent font-medium">15 minutes</span>.
          </p>
        </motion.div>

        {/* Connection Diagram */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.3 }}
        >
          <SpotlightCard className="p-8 md:p-12">
            <div className="flex flex-col md:flex-row items-center justify-center gap-8 md:gap-12">
              {/* Camera Icons */}
              <div className="flex gap-6 md:gap-8">
                {cameraIcons.map(({ Icon, label }, index) => (
                  <motion.div
                    key={label}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.5, delay: 0.2 + index * 0.1 }}
                    className="flex flex-col items-center gap-3"
                  >
                    <div className="
                      w-16 h-16 rounded-xl flex items-center justify-center
                      bg-white/[0.03] border border-white/[0.06]
                    ">
                      <Icon className="w-8 h-8 text-foreground-muted/50" />
                    </div>
                    <span className="text-xs text-foreground-muted/60">{label}</span>
                  </motion.div>
                ))}
              </div>

              {/* Animated Connection Lines */}
              <div className="relative w-24 md:w-32 h-16 flex items-center">
                <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 60">
                  {/* Dashed lines converging */}
                  {[0, 1, 2].map((i) => (
                    <motion.line
                      key={i}
                      x1="0"
                      y1={15 + i * 15}
                      x2="100"
                      y2="30"
                      stroke={COLORS.accent}
                      strokeWidth="1.5"
                      strokeDasharray="4 4"
                      strokeOpacity="0.5"
                      initial={{ pathLength: 0 }}
                      whileInView={{ pathLength: 1 }}
                      viewport={{ once: true }}
                      transition={{ duration: 1, delay: 0.5 + i * 0.2 }}
                    />
                  ))}
                  {/* Animated data flow dots */}
                  {[0, 1, 2].map((i) => (
                    <motion.circle
                      key={`dot-${i}`}
                      r="3"
                      fill={COLORS.accent}
                      initial={{ opacity: 0 }}
                      animate={{ 
                        opacity: [0, 1, 1, 0],
                        cx: [0, 50, 100],
                        cy: [15 + i * 15, 22.5 + i * 7.5, 30]
                      }}
                      transition={{ 
                        duration: 2, 
                        delay: i * 0.3, 
                        repeat: Infinity,
                        repeatDelay: 1
                      }}
                    />
                  ))}
                </svg>
              </div>

              {/* Central Shield */}
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: 0.8 }}
                className="relative"
              >
                <div className="
                  w-24 h-24 rounded-2xl flex items-center justify-center
                  bg-gradient-to-br from-accent/20 to-accent/5
                  border border-accent/30
                  shadow-[0_0_40px_rgba(94,106,210,0.3)]
                ">
                  <Shield className="w-12 h-12 text-accent" />
                </div>
                {/* Pulsing glow */}
                <motion.div
                  className="absolute inset-0 rounded-2xl"
                  style={{ boxShadow: `0 0 60px 10px ${COLORS.accentGlow}` }}
                  animate={{ opacity: [0.3, 0.6, 0.3] }}
                  transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                />
              </motion.div>

              {/* Processing Status */}
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: 1 }}
                className="flex flex-col gap-2 text-left"
              >
                <div className="flex items-center gap-2">
                  <motion.div
                    className="w-2 h-2 rounded-full bg-accent"
                    animate={{ opacity: [1, 0.3, 1] }}
                    transition={{ duration: 1, repeat: Infinity }}
                  />
                  <span className="text-sm font-mono text-foreground-muted">Processing...</span>
                </div>
                <motion.div
                  initial={{ opacity: 0 }}
                  whileInView={{ opacity: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: 1.5 }}
                  className="flex items-center gap-2"
                >
                  <div className="w-2 h-2 rounded-full bg-status-success" />
                  <span className="text-sm font-mono text-status-success font-medium">Secured</span>
                </motion.div>
              </motion.div>
            </div>
          </SpotlightCard>
        </motion.div>
      </div>
    </section>
  )
}

// ============================================================================
// SHIFT REPORT SECTION - "The Shift Report"
// ============================================================================
function ShiftReportSection() {
  const timelineEvents = [
    { time: '02:14 AM', event: 'Agitation Spike (Resolved)', status: 'warning' },
    { time: '04:30 AM', event: 'SpO2 Dip (Duration: 45s)', status: 'error' },
    { time: '05:45 AM', event: 'Patient Repositioned', status: 'success' },
  ]

  return (
    <section className="relative py-24 md:py-32">
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
          {/* Left - Text Content */}
          <motion.div
            initial={{ opacity: 0, x: -24 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          >
            <span className="text-xs font-mono uppercase tracking-widest text-accent mb-4 block">
              Automated Documentation
            </span>
            <h2 className="text-3xl md:text-4xl font-semibold tracking-tight mb-6 bg-gradient-to-b from-white to-white/70 bg-clip-text text-transparent">
              Paperwork, handled.
            </h2>
            <p className="text-lg text-foreground-muted leading-relaxed mb-6">
              Automated event logging generates a precise timeline of patient activity for the 
              morning handover, saving nurses <span className="text-accent font-medium">2 hours of 
              documentation time</span>.
            </p>
            <ul className="space-y-3">
              {[
                'Real-time event capture and classification',
                'Exportable shift summaries',
                'EPIC & Cerner EMR integration ready',
                'Timestamped with video evidence links'
              ].map((item, i) => (
                <li key={i} className="flex items-center gap-3 text-foreground-muted">
                  <div className="w-1.5 h-1.5 rounded-full bg-accent" />
                  {item}
                </li>
              ))}
            </ul>
          </motion.div>

          {/* Right - 3D Floating Report Card */}
          <motion.div
            initial={{ opacity: 0, x: 24 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="relative perspective-1000"
          >
            <motion.div
              className="relative"
              animate={{ 
                rotateY: [0, 3, 0, -3, 0],
                rotateX: [0, -2, 0, 2, 0]
              }}
              transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
              style={{ 
                transformStyle: 'preserve-3d',
                transform: 'perspective(1000px) rotateY(-5deg) rotateX(5deg)'
              }}
            >
              <SpotlightCard className="p-6 relative overflow-hidden">
                {/* Report Header */}
                <div className="flex items-center justify-between mb-6 pb-4 border-b border-white/[0.06]">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-accent/10 border border-accent/20 flex items-center justify-center">
                      <FileText className="w-5 h-5 text-accent" />
                    </div>
                    <div>
                      <h4 className="font-semibold text-foreground">Shift Summary: Bed 4</h4>
                      <p className="text-xs text-foreground-muted">Night Shift • 10:00 PM - 6:00 AM</p>
                    </div>
                  </div>
                  <div className="text-xs font-mono text-foreground-muted bg-white/[0.03] px-2 py-1 rounded">
                    AUTO-GENERATED
                  </div>
                </div>

                {/* Timeline */}
                <div className="space-y-4 mb-6">
                  {timelineEvents.map((event, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -10 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.4, delay: 0.5 + index * 0.15 }}
                      className="flex items-start gap-4"
                    >
                      <div className="flex flex-col items-center">
                        <div className={`
                          w-3 h-3 rounded-full
                          ${event.status === 'warning' ? 'bg-status-warning' : ''}
                          ${event.status === 'error' ? 'bg-status-error' : ''}
                          ${event.status === 'success' ? 'bg-status-success' : ''}
                        `} />
                        {index < timelineEvents.length - 1 && (
                          <div className="w-px h-8 bg-white/[0.1] mt-1" />
                        )}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <Clock className="w-3.5 h-3.5 text-foreground-muted" />
                          <span className="text-xs font-mono text-foreground-muted">{event.time}</span>
                        </div>
                        <p className="text-sm text-foreground mt-1">{event.event}</p>
                      </div>
                    </motion.div>
                  ))}
                </div>

                {/* Export Button */}
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="
                    w-full py-3 rounded-lg
                    bg-gradient-to-r from-accent/20 to-accent/10
                    border border-accent/30
                    text-accent font-medium text-sm
                    flex items-center justify-center gap-2
                    hover:border-accent/50 hover:from-accent/25 hover:to-accent/15
                    transition-all duration-300
                  "
                >
                  <Download className="w-4 h-4" />
                  Export to EMR (EPIC/Cerner)
                </motion.button>

                {/* 3D Shadow layers */}
                <div className="absolute -bottom-2 -right-2 -left-2 h-4 bg-gradient-to-t from-black/20 to-transparent rounded-b-2xl -z-10" style={{ transform: 'translateZ(-10px)' }} />
              </SpotlightCard>
            </motion.div>

            {/* Decorative floating elements */}
            <motion.div
              animate={{ y: [0, -10, 0] }}
              transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
              className="absolute -top-4 -right-4 w-8 h-8 rounded-lg bg-accent/10 border border-accent/20"
              style={{ transform: 'translateZ(20px)' }}
            />
            <motion.div
              animate={{ y: [0, 8, 0] }}
              transition={{ duration: 5, repeat: Infinity, ease: "easeInOut", delay: 1 }}
              className="absolute -bottom-4 -left-4 w-6 h-6 rounded-full bg-accent/5 border border-accent/10"
              style={{ transform: 'translateZ(30px)' }}
            />
          </motion.div>
        </div>
      </div>
    </section>
  )
}

// ============================================================================
// TEAM SECTION - "The Minds Behind the Mission"
// ============================================================================
// Team Member Card with Spotlight Effect
interface TeamCardProps {
  member: {
    name: string
    role: string
    description: string
    linkedin: string
    avatar: string
    imagePosition?: string
  }
  index: number
}

function TeamCard({ member, index }: TeamCardProps) {
  const cardRef = useRef<HTMLDivElement>(null)

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return
    const rect = cardRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    cardRef.current.style.setProperty('--mouse-x', `${x}px`)
    cardRef.current.style.setProperty('--mouse-y', `${y}px`)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay: index * 0.1, ease: [0.16, 1, 0.3, 1] }}
      className="group h-full"
    >
      <div
        ref={cardRef}
        onMouseMove={handleMouseMove}
        className="
          team-spotlight-card
          relative h-[320px] p-6 rounded-2xl
          bg-white/[0.05] backdrop-blur-xl
          border border-white/[0.06]
          transition-all duration-300
          hover:scale-[1.02] hover:bg-white/[0.08]
          hover:border-accent/40
          hover:shadow-[0_0_60px_rgba(94,106,210,0.2)]
          overflow-hidden
        "
      >
        {/* Mouse-tracking spotlight gradient */}
        <div 
          className="
            pointer-events-none absolute inset-0 opacity-0 group-hover:opacity-100
            transition-opacity duration-300
          "
          style={{
            background: 'radial-gradient(350px circle at var(--mouse-x, 50%) var(--mouse-y, 50%), rgba(94,106,210,0.15), transparent 60%)'
          }}
        />
        
        {/* Glow effect on hover */}
        <div className="absolute inset-0 rounded-2xl bg-gradient-to-b from-accent/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
        
        {/* Avatar */}
        <div className="relative flex justify-center mb-4">
          <div className="relative">
            <div className="w-20 h-20 rounded-full border-2 border-accent/40 group-hover:border-accent transition-all duration-300 overflow-hidden bg-background-elevated shadow-[0_0_20px_rgba(94,106,210,0.1)] group-hover:shadow-[0_0_30px_rgba(94,106,210,0.3)]">
              <img
                src={member.avatar}
                alt={member.name}
                className="w-full h-full object-cover"
                style={{ objectPosition: member.imagePosition || 'top' }}
              />
            </div>
            {/* Glow ring on hover */}
            <div className="absolute inset-0 rounded-full bg-accent/30 blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10 scale-110" />
          </div>
        </div>

        {/* Info */}
        <div className="relative text-center flex flex-col h-[calc(100%-96px)]">
          <h3 className="text-lg font-semibold text-foreground mb-1 group-hover:text-white transition-colors">
            {member.name}
          </h3>
          <p className="text-xs font-medium text-accent mb-2">
            {member.role}
          </p>
          
          {/* Description - shows on hover with fade */}
          <p className="text-xs text-foreground-muted leading-relaxed mb-4 opacity-60 group-hover:opacity-100 transition-opacity duration-300 flex-grow">
            {member.description}
          </p>

          {/* Social Links */}
          <div className="flex justify-center gap-3 mt-auto">
            <a
              href={member.linkedin}
              target="_blank"
              rel="noopener noreferrer"
              className="
                p-2.5 rounded-xl
                bg-white/[0.03] border border-white/[0.06]
                text-foreground-muted
                hover:text-accent hover:bg-accent/10 hover:border-accent/30
                transition-all duration-200
                group/icon
              "
              aria-label={`${member.name}'s LinkedIn`}
            >
              <Linkedin className="w-4 h-4 group-hover/icon:scale-110 transition-transform" />
            </a>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

function TeamSection() {
  const teamMembers = [
    {
      name: 'Kailas V Sharji',
      role: 'Team Lead | ML Training',
      description: 'Leads the team and drives ML model development, training, and optimization for real-time patient monitoring.',
      linkedin: 'https://www.linkedin.com/in/kailas-v-sharji-b78861235/',
      avatar: '/team/kailas.jpg'
    },
    {
      name: 'Jeff Joseph',
      role: 'Full Stack Developer',
      description: 'Architects and builds the complete application stack, from React frontend to Python backend APIs.',
      linkedin: 'https://www.linkedin.com/in/jeff-joseph1/',
      avatar: '/team/jeff.jpg',
      imagePosition: 'center'
    },
    {
      name: 'Irine Milton',
      role: 'Ideation & Strategy',
      description: 'Shapes product vision, user experience strategy, and ensures clinical relevance of all features.',
      linkedin: 'https://www.linkedin.com/in/irinemilton/',
      avatar: '/team/irine.jpg'
    },
    {
      name: 'Lee Paul Anto',
      role: 'Technical Support & Research',
      description: 'Conducts research on ICU protocols, validates technical implementations, and ensures system reliability.',
      linkedin: 'https://www.linkedin.com/in/lee-paul-anto-57ba7b326/',
      avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=Lee'
    }
  ]

  return (
    <section id="team" className="relative py-24 md:py-32">
      {/* Top gradient line */}
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          className="text-center mb-16"
        >
          <span className="inline-block text-xs font-mono uppercase tracking-widest text-accent mb-4">
            Our Team
          </span>
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-semibold tracking-tight">
            <span className="bg-gradient-to-b from-white via-white/95 to-white/70 bg-clip-text text-transparent">
              The Minds Behind the Mission.
            </span>
          </h2>
        </motion.div>

        {/* Team Grid - All cards same size */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {teamMembers.map((member, index) => (
            <TeamCard key={member.name} member={member} index={index} />
          ))}
        </div>
      </div>
    </section>
  )
}

// ============================================================================
// TECH STACK SECTION
// ============================================================================
function TechStackSection() {
  const techStack = [
    { 
      name: 'Python', 
      icon: (
        <svg viewBox="0 0 24 24" className="w-6 h-6" fill="currentColor">
          <path d="M14.25.18l.9.2.73.26.59.3.45.32.34.34.25.34.16.33.1.3.04.26.02.2-.01.13V8.5l-.05.63-.13.55-.21.46-.26.38-.3.31-.33.25-.35.19-.35.14-.33.1-.3.07-.26.04-.21.02H8.77l-.69.05-.59.14-.5.22-.41.27-.33.32-.27.35-.2.36-.15.37-.1.35-.07.32-.04.27-.02.21v3.06H3.17l-.21-.03-.28-.07-.32-.12-.35-.18-.36-.26-.36-.36-.35-.46-.32-.59-.28-.73-.21-.88-.14-1.05-.05-1.23.06-1.22.16-1.04.24-.87.32-.71.36-.57.4-.44.42-.33.42-.24.4-.16.36-.1.32-.05.24-.01h.16l.06.01h8.16v-.83H6.18l-.01-2.75-.02-.37.05-.34.11-.31.17-.28.25-.26.31-.23.38-.2.44-.18.51-.15.58-.12.64-.1.71-.06.77-.04.84-.02 1.27.05zm-6.3 1.98l-.23.33-.08.41.08.41.23.34.33.22.41.09.41-.09.33-.22.23-.34.08-.41-.08-.41-.23-.33-.33-.22-.41-.09-.41.09zm13.09 3.95l.28.06.32.12.35.18.36.27.36.35.35.47.32.59.28.73.21.88.14 1.04.05 1.23-.06 1.23-.16 1.04-.24.86-.32.71-.36.57-.4.45-.42.33-.42.24-.4.16-.36.09-.32.05-.24.02-.16-.01h-8.22v.82h5.84l.01 2.76.02.36-.05.34-.11.31-.17.29-.25.25-.31.24-.38.2-.44.17-.51.15-.58.13-.64.09-.71.07-.77.04-.84.01-1.27-.04-1.07-.14-.9-.2-.73-.25-.59-.3-.45-.33-.34-.34-.25-.34-.16-.33-.1-.3-.04-.25-.02-.2.01-.13v-5.34l.05-.64.13-.54.21-.46.26-.38.3-.32.33-.24.35-.2.35-.14.33-.1.3-.06.26-.04.21-.02.13-.01h5.84l.69-.05.59-.14.5-.21.41-.28.33-.32.27-.35.2-.36.15-.36.1-.35.07-.32.04-.28.02-.21V6.07h2.09l.14.01zm-6.47 14.25l-.23.33-.08.41.08.41.23.33.33.23.41.08.41-.08.33-.23.23-.33.08-.41-.08-.41-.23-.33-.33-.23-.41-.08-.41.08z"/>
        </svg>
      )
    },
    { 
      name: 'TensorFlow', 
      icon: (
        <svg viewBox="0 0 24 24" className="w-6 h-6" fill="currentColor">
          <path d="M1.292 5.856L11.54 0v24l-4.095-2.378V7.603l-6.168 3.564.015-5.31zm21.416 5.393l-.014-5.31-10.249-5.94v4.69l6.154 3.563v8.744l-6.154 3.566V24l10.249-5.94.014-6.81z"/>
        </svg>
      )
    },
    { 
      name: 'MediaPipe', 
      icon: <Video className="w-6 h-6" />
    },
    { 
      name: 'Streamlit', 
      icon: (
        <svg viewBox="0 0 24 24" className="w-6 h-6" fill="currentColor">
          <path d="M12.052 0L1.631 5.99l5.1 2.874L12.052 5.7l5.322 3.164 5.1-2.874L12.053 0zm-.208 9.654L6.73 12.61v5.696l5.114 2.908 5.113-2.908V12.61l-5.113-2.956z"/>
        </svg>
      )
    },
    { 
      name: 'OpenCV', 
      icon: <Eye className="w-6 h-6" />
    },
  ]

  return (
    <section id="tech" className="relative py-16 md:py-24">
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          className="text-center mb-12"
        >
          <span className="text-xs font-mono uppercase tracking-widest text-foreground-muted">
            Powered By
          </span>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="flex flex-nowrap justify-center items-center gap-4 md:gap-8 lg:gap-12 overflow-x-auto pb-2"
        >
          {techStack.map((tech, index) => (
            <motion.div
              key={tech.name}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: 0.1 * index }}
              className="
                group flex items-center gap-3 px-6 py-3 rounded-xl
                bg-white/[0.02] border border-white/[0.04]
                hover:bg-white/[0.05] hover:border-white/[0.08]
                transition-all duration-300
                opacity-60 hover:opacity-100
              "
            >
              <span className="text-accent group-hover:text-accent-bright transition-colors">
                {tech.icon}
              </span>
              <span className="text-sm font-medium text-foreground-muted group-hover:text-foreground transition-colors">
                {tech.name}
              </span>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  )
}

// ============================================================================
// FOOTER
// ============================================================================
function Footer() {
  return (
    <footer className="relative py-12 border-t border-white/[0.06]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row items-center justify-center gap-6">
          <p className="text-sm text-foreground-muted">
            © 2026 ICU Guardian
          </p>
        </div>
      </div>
    </footer>
  )
}

// ============================================================================
// MAIN LANDING PAGE
// ============================================================================
export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background-base text-foreground antialiased overflow-x-hidden">
      {/* Ambient Background - Same as Dashboard */}
      <AmbientBackground />

      {/* Content */}
      <div className="relative z-10">
        <Navbar />
        <main>
          <HeroSection />
          <SecureCoreVisualization />
          <FeaturesSection />
          <DrAISection />
          <AlarmFatigueSection />
          <InfrastructureSection />
          <ShiftReportSection />
          <CapabilitiesBentoGrid />
          <ZeroTrustSecurity />
          <DeploymentRoadmap />
          <ImpactStatistics />
          <TeamSection />
          <TechStackSection />
        </main>
        <Footer />
      </div>

      {/* Scroll to Top Button */}
      <ScrollToTop />
    </div>
  )
}
