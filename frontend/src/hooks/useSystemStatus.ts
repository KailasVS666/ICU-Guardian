import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000',
})

export function useSystemStatus() {
  const { data: systemStatus } = useQuery({
    queryKey: ['systemStatus'],
    queryFn: async () => {
      const { data } = await api.get('/api/system/status')
      return data
    },
    refetchInterval: 2000,
  })

  return { systemStatus }
}

export function useVisionControl() {
  const queryClient = useQueryClient()

  const startVision = useMutation({
    mutationFn: async () => {
      const response = await api.post('/api/vision/start')
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['systemStatus'] })
      console.log('✅ Vision started')
    },
    onError: (error) => {
      console.error('❌ Failed to start vision:', error)
    },
  })

  const stopVision = useMutation({
    mutationFn: async () => {
      const response = await api.post('/api/vision/stop')
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['systemStatus'] })
      console.log('✅ Vision stopped')
    },
    onError: (error) => {
      console.error('❌ Failed to stop vision:', error)
    },
  })

  return { startVision, stopVision }
}

export function useMonitoringControl() {
  const queryClient = useQueryClient()

  const startMonitoring = useMutation({
    mutationFn: async () => {
      const response = await api.post('/api/monitoring/start')
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['systemStatus'] })
      console.log('✅ Monitoring started')
    },
    onError: (error) => {
      console.error('❌ Failed to start monitoring:', error)
    },
  })

  const stopMonitoring = useMutation({
    mutationFn: async () => {
      const response = await api.post('/api/monitoring/stop')
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['systemStatus'] })
      console.log('✅ Monitoring stopped')
    },
    onError: (error) => {
      console.error('❌ Failed to stop monitoring:', error)
    },
  })

  return { startMonitoring, stopMonitoring }
}
