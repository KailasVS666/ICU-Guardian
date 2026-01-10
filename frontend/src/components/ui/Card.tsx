import { useRef, MouseEvent, ReactNode } from 'react'
import { clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

interface CardProps {
  children: ReactNode
  className?: string
  spotlight?: boolean
  variant?: 'default' | 'glass' | 'elevated'
  padding?: 'none' | 'sm' | 'md' | 'lg'
  hover?: boolean
}

function cn(...inputs: (string | undefined | null | boolean)[]) {
  return twMerge(clsx(inputs))
}

export default function Card({ 
  children, 
  className, 
  spotlight = true, 
  variant = 'default',
  padding = 'md',
  hover = true
}: CardProps) {
  const cardRef = useRef<HTMLDivElement>(null)

  const handleMouseMove = (e: MouseEvent<HTMLDivElement>) => {
    if (!spotlight || !cardRef.current) return
    
    const rect = cardRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    
    cardRef.current.style.setProperty('--mouse-x', `${x}px`)
    cardRef.current.style.setProperty('--mouse-y', `${y}px`)
  }

  const paddingClasses = {
    none: '',
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8'
  }

  const variantClasses = {
    default: 'glass-card',
    glass: 'glass-card backdrop-blur-xl',
    elevated: 'glass-card bg-background-elevated'
  }

  return (
    <div
      ref={cardRef}
      onMouseMove={handleMouseMove}
      className={cn(
        variantClasses[variant],
        spotlight && 'spotlight-card',
        hover && 'hover:transform hover:-translate-y-1',
        paddingClasses[padding],
        className
      )}
    >
      {children}
    </div>
  )
}

// Sub-components for semantic structure
Card.Header = function CardHeader({ 
  children, 
  className 
}: { 
  children: ReactNode
  className?: string 
}) {
  return (
    <div className={cn('mb-4', className)}>
      {children}
    </div>
  )
}

Card.Title = function CardTitle({ 
  children, 
  className 
}: { 
  children: ReactNode
  className?: string 
}) {
  return (
    <h3 className={cn(
      'text-lg font-semibold tracking-tight text-foreground',
      className
    )}>
      {children}
    </h3>
  )
}

Card.Description = function CardDescription({ 
  children, 
  className 
}: { 
  children: ReactNode
  className?: string 
}) {
  return (
    <p className={cn(
      'text-sm text-foreground-muted leading-relaxed',
      className
    )}>
      {children}
    </p>
  )
}

Card.Content = function CardContent({ 
  children, 
  className 
}: { 
  children: ReactNode
  className?: string 
}) {
  return (
    <div className={cn('', className)}>
      {children}
    </div>
  )
}
