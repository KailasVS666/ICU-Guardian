/**
 * AmbientBackground - Animated gradient blobs for cinematic lighting
 * 
 * This component renders the floating gradient shapes that create
 * the "ambient lighting" effect characteristic of the Linear design system.
 * 
 * Features:
 * - Multiple layered, floating gradient shapes for ambient "light pools"
 * - Respects prefers-reduced-motion for accessibility
 * - Uses CSS classes from index.css for animations
 */

export default function AmbientBackground() {
  return (
    <div 
      className="fixed inset-0 overflow-hidden pointer-events-none z-0" 
      aria-hidden="true"
    >
      {/* Primary blob - Top center, indigo accent (900×700px) */}
      <div 
        className="
          absolute rounded-full
          w-[900px] h-[700px]
          -top-[20%] left-1/2 -translate-x-1/2
          bg-[radial-gradient(circle,rgba(94,106,210,0.25)_0%,transparent_70%)]
          blur-[150px]
          motion-safe:animate-[floatPrimary_8s_ease-in-out_infinite]
        "
      />
      
      {/* Secondary blob - Left side, purple tint (600×800px) */}
      <div 
        className="
          absolute rounded-full
          w-[600px] h-[800px]
          top-[20%] -left-[10%]
          bg-[radial-gradient(circle,rgba(124,58,237,0.15)_0%,transparent_70%)]
          blur-[120px]
          motion-safe:animate-[float_10s_ease-in-out_infinite_-3s]
        "
      />
      
      {/* Tertiary blob - Right side, blue tint (500×700px) */}
      <div 
        className="
          absolute rounded-full
          w-[500px] h-[700px]
          top-[40%] -right-[15%]
          bg-[radial-gradient(circle,rgba(59,130,246,0.12)_0%,transparent_70%)]
          blur-[100px]
          motion-safe:animate-[float_12s_ease-in-out_infinite_-5s]
        "
      />
      
      {/* Bottom accent blob - Pulsing glow (800×600px) */}
      <div 
        className="
          absolute rounded-full
          w-[800px] h-[600px]
          -bottom-[30%] left-[30%]
          bg-[radial-gradient(circle,rgba(94,106,210,0.1)_0%,transparent_70%)]
          blur-[150px]
          motion-safe:animate-[pulseGlow_3s_ease-in-out_infinite]
        "
      />
      
      {/* Medical/ICU specific - Subtle red accent for alerts context (400×400px) */}
      <div 
        className="
          absolute rounded-full
          w-[400px] h-[400px]
          top-[60%] right-[10%]
          bg-[radial-gradient(circle,rgba(239,68,68,0.08)_0%,transparent_70%)]
          blur-[100px]
          motion-safe:animate-[float_15s_ease-in-out_infinite_-7s]
        "
      />
    </div>
  )
}
