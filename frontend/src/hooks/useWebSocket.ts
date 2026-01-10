import { useState, useEffect, useRef } from 'react'

interface VitalsData {
  vitals: {
    HR_Avg: number
    SpO2_Min: number
    Sleep_Score: number
    RASS_Score: number
  }
  prediction: {
    risk_level: number
    probability: number
    risk_category: string
  }
  timestamp: string
}

interface AlertData {
  pillars: {
    [key: string]: {
      active: boolean
      count: number
      last_alert?: string
    }
  }
}

export function useWebSocket() {
  const [isConnected, setIsConnected] = useState(false)
  const [vitals, setVitals] = useState<VitalsData | null>(null)
  const [alerts, setAlerts] = useState<AlertData | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>()
  const retryAttemptRef = useRef(0)

  const connect = () => {
    const ws = new WebSocket('ws://localhost:8000/ws')

    ws.onopen = () => {
      console.log('✅ WebSocket connected')
      setIsConnected(true)
      retryAttemptRef.current = 0
    }

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data)
        
        switch (message.type) {
          case 'vitals_update':
            setVitals(message.data)
            break
          case 'alerts_update':
            setAlerts(message.data)
            break
          case 'connection_established':
            console.log('✅ Connected to ICU Guardian')
            break
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error)
      }
    }

    ws.onclose = () => {
      console.log('❌ WebSocket disconnected')
      setIsConnected(false)
      retryAttemptRef.current += 1
      const delay = Math.min(30000, 1000 * 2 ** (retryAttemptRef.current - 1))
      
      reconnectTimeoutRef.current = setTimeout(() => {
        console.log(`🔄 Attempting to reconnect (backoff ${delay}ms)...`)
        connect()
      }, delay)
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    wsRef.current = ws
  }

  useEffect(() => {
    connect()

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  return { isConnected, vitals, alerts }
}
