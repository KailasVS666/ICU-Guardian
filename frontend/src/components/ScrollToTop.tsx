/**
 * Scroll to Top Button - ICU Guardian
 * 
 * Floating action button that appears on scroll
 * Smooth scroll back to top with professional animations
 */

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowUp } from 'lucide-react'

export function ScrollToTop() {
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    const toggleVisibility = () => {
      if (window.scrollY > 400) {
        setIsVisible(true)
      } else {
        setIsVisible(false)
      }
    }

    window.addEventListener('scroll', toggleVisibility)

    return () => {
      window.removeEventListener('scroll', toggleVisibility)
    }
  }, [])

  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    })
  }

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.button
          initial={{ opacity: 0, scale: 0.8, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.8, y: 20 }}
          transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
          onClick={scrollToTop}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          className="
            fixed bottom-8 right-8 z-50
            w-12 h-12 rounded-xl
            bg-gradient-to-br from-accent/90 to-accent/70
            backdrop-blur-xl
            border border-accent/50
            flex items-center justify-center
            text-white
            shadow-lg shadow-accent/20
            hover:shadow-xl hover:shadow-accent/30
            transition-shadow duration-300
            cursor-pointer
            group
          "
          aria-label="Scroll to top"
        >
          <ArrowUp 
            className="w-5 h-5 group-hover:-translate-y-0.5 transition-transform duration-300" 
            strokeWidth={2.5}
          />
        </motion.button>
      )}
    </AnimatePresence>
  )
}

export default ScrollToTop
