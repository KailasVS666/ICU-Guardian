import { ReactNode } from 'react'
import { clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

type BadgeVariant = 'default' | 'success' | 'warning' | 'error' | 'info'
type BadgeSize = 'sm' | 'md'

interface BadgeProps {
  children: ReactNode
  variant?: BadgeVariant
  size?: BadgeSize
  icon?: ReactNode
  pulse?: boolean
  className?: string
}

function cn(...inputs: (string | undefined | null | boolean)[]) {
  return twMerge(clsx(inputs))
}

export default function Badge({
  children,
  variant = 'default',
  size = 'md',
  icon,
  pulse = false,
  className
}: BadgeProps) {
  const baseStyles = `
    inline-flex items-center gap-1.5
    font-semibold uppercase tracking-wider
    rounded-full
    border
  `

  const variantStyles = {
    default: 'bg-white/[0.05] text-foreground-muted border-white/10',
    success: 'bg-green-500/15 text-green-400 border-green-500/30',
    warning: 'bg-yellow-500/15 text-yellow-400 border-yellow-500/30',
    error: 'bg-red-500/15 text-red-400 border-red-500/30',
    info: 'bg-accent/15 text-accent-bright border-accent/30'
  }

  const sizeStyles = {
    sm: 'px-2 py-0.5 text-[10px]',
    md: 'px-3 py-1 text-xs'
  }

  return (
    <span className={cn(
      baseStyles,
      variantStyles[variant],
      sizeStyles[size],
      className
    )}>
      {pulse && (
        <span className="relative flex h-2 w-2">
          <span className={cn(
            'animate-ping absolute inline-flex h-full w-full rounded-full opacity-75',
            variant === 'success' && 'bg-green-400',
            variant === 'warning' && 'bg-yellow-400',
            variant === 'error' && 'bg-red-400',
            variant === 'info' && 'bg-accent',
            variant === 'default' && 'bg-foreground-muted'
          )} />
          <span className={cn(
            'relative inline-flex rounded-full h-2 w-2',
            variant === 'success' && 'bg-green-400',
            variant === 'warning' && 'bg-yellow-400',
            variant === 'error' && 'bg-red-400',
            variant === 'info' && 'bg-accent',
            variant === 'default' && 'bg-foreground-muted'
          )} />
        </span>
      )}
      {icon}
      {children}
    </span>
  )
}
