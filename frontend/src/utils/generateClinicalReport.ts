/**
 * Clinical Report PDF Generator
 * 
 * Generates a professional A4 PDF clinical report using jsPDF
 * Styled for hospital/medical professional use
 */

import jsPDF from 'jspdf'

interface VitalsData {
  hr: number
  spo2: number
  rass: number
  sleepScore: number
  riskProbability: number
}

interface AlertData {
  timestamp: string
  pillar: string
  message: string
  severity: string
}

interface ReportData {
  vitals: VitalsData
  alerts: AlertData[]
  hrHistory: number[]
  spo2History: number[]
  aiMessage: string
  uptime: string
}

// Color constants matching the design system
const COLORS = {
  accent: [94, 106, 210] as [number, number, number],      // #5E6AD2
  headerGray: [60, 60, 60] as [number, number, number],
  textColor: [30, 30, 30] as [number, number, number],
  lightGray: [180, 180, 180] as [number, number, number],
  criticalRed: [220, 53, 69] as [number, number, number],
  warningOrange: [255, 152, 0] as [number, number, number],
  successGreen: [40, 167, 69] as [number, number, number],
  bgGray: [245, 245, 245] as [number, number, number],
  lineGray: [200, 200, 200] as [number, number, number],
  graphBlue: [52, 152, 219] as [number, number, number],
}

export function generateClinicalReport(data: ReportData): void {
  const pdf = new jsPDF('p', 'mm', 'a4')
  const pageWidth = pdf.internal.pageSize.getWidth()
  
  let yPos = 15

  // ==================== HEADER ====================
  // ICU Guardian Logo/Title
  pdf.setFont('helvetica', 'bold')
  pdf.setFontSize(18)
  pdf.setTextColor(...COLORS.accent)
  pdf.text('ICU GUARDIAN™', 15, yPos)
  
  // Timestamp on right
  pdf.setFont('helvetica', 'normal')
  pdf.setFontSize(10)
  pdf.setTextColor(...COLORS.headerGray)
  const timestamp = new Date().toLocaleString('en-US', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
  pdf.text(`Generated: ${timestamp}`, pageWidth - 15, yPos, { align: 'right' })
  
  yPos += 7
  
  // Sub-header
  pdf.setFont('helvetica', 'italic')
  pdf.setFontSize(10)
  pdf.setTextColor(...COLORS.lightGray)
  pdf.text('Confidential Clinical Summary | Neuro-Trauma Unit', 15, yPos)
  
  yPos += 5
  
  // Separator line
  pdf.setDrawColor(...COLORS.accent)
  pdf.setLineWidth(0.5)
  pdf.line(15, yPos, pageWidth - 15, yPos)
  
  yPos += 12

  // ==================== VITALS TABLE ====================
  // Section title
  pdf.setFont('helvetica', 'bold')
  pdf.setFontSize(12)
  pdf.setTextColor(...COLORS.headerGray)
  pdf.setFillColor(...COLORS.bgGray)
  pdf.rect(15, yPos - 5, pageWidth - 30, 8, 'F')
  pdf.text('PATIENT VITALS SNAPSHOT', 17, yPos)
  
  yPos += 10
  
  // Table header
  const colWidths = [50, 45, 50, 45]
  const headers = ['Parameter', 'Value', 'Parameter', 'Value']
  let xPos = 15
  
  pdf.setFont('helvetica', 'bold')
  pdf.setFontSize(10)
  pdf.setTextColor(255, 255, 255)
  pdf.setFillColor(...COLORS.headerGray)
  
  headers.forEach((header, i) => {
    pdf.rect(xPos, yPos - 5, colWidths[i], 8, 'F')
    pdf.text(header, xPos + colWidths[i] / 2, yPos, { align: 'center' })
    xPos += colWidths[i]
  })
  
  yPos += 8
  
  // Table data - Row 1
  pdf.setFont('helvetica', 'normal')
  pdf.setTextColor(...COLORS.textColor)
  pdf.setDrawColor(...COLORS.lineGray)
  
  xPos = 15
  const row1 = [
    { label: 'Average Heart Rate', value: `${data.vitals.hr} BPM` },
    { label: 'Minimum SpO2', value: `${data.vitals.spo2}%` }
  ]
  
  row1.forEach((cell, i) => {
    pdf.rect(xPos, yPos - 5, colWidths[i * 2], 8)
    pdf.text(cell.label, xPos + 2, yPos)
    pdf.rect(xPos + colWidths[i * 2], yPos - 5, colWidths[i * 2 + 1], 8)
    pdf.text(cell.value, xPos + colWidths[i * 2] + colWidths[i * 2 + 1] / 2, yPos, { align: 'center' })
    xPos += colWidths[i * 2] + colWidths[i * 2 + 1]
  })
  
  yPos += 8
  
  // Table data - Row 2
  xPos = 15
  pdf.rect(xPos, yPos - 5, colWidths[0], 8)
  pdf.text('RASS Score', xPos + 2, yPos)
  pdf.rect(xPos + colWidths[0], yPos - 5, colWidths[1], 8)
  pdf.text(String(data.vitals.rass), xPos + colWidths[0] + colWidths[1] / 2, yPos, { align: 'center' })
  
  xPos += colWidths[0] + colWidths[1]
  pdf.rect(xPos, yPos - 5, colWidths[2], 8)
  pdf.text('AI Risk Probability', xPos + 2, yPos)
  pdf.rect(xPos + colWidths[2], yPos - 5, colWidths[3], 8)
  
  // Color code risk
  const risk = data.vitals.riskProbability
  if (risk > 0.6) {
    pdf.setTextColor(...COLORS.criticalRed)
  } else if (risk > 0.4) {
    pdf.setTextColor(...COLORS.warningOrange)
  } else {
    pdf.setTextColor(...COLORS.successGreen)
  }
  pdf.text(`${(risk * 100).toFixed(0)}%`, xPos + colWidths[2] + colWidths[3] / 2, yPos, { align: 'center' })
  pdf.setTextColor(...COLORS.textColor)
  
  yPos += 15

  // ==================== VITALS TREND GRAPH ====================
  // Section title
  pdf.setFont('helvetica', 'bold')
  pdf.setFontSize(12)
  pdf.setTextColor(...COLORS.headerGray)
  pdf.setFillColor(...COLORS.bgGray)
  pdf.rect(15, yPos - 5, pageWidth - 30, 8, 'F')
  pdf.text('PHYSIOLOGICAL TREND ANALYSIS (Last 30 mins)', 17, yPos)
  
  yPos += 10
  
  // Graph dimensions
  const graphX = 15
  const graphY = yPos
  const graphWidth = pageWidth - 30
  const graphHeight = 45
  
  // Draw graph background
  pdf.setFillColor(250, 250, 250)
  pdf.setDrawColor(...COLORS.lineGray)
  pdf.rect(graphX, graphY, graphWidth, graphHeight, 'DF')
  
  // Draw grid lines
  pdf.setDrawColor(230, 230, 230)
  for (let i = 1; i < 5; i++) {
    const yLine = graphY + (graphHeight / 5) * i
    pdf.line(graphX, yLine, graphX + graphWidth, yLine)
  }
  
  // Draw HR line (accent color)
  if (data.hrHistory.length >= 2) {
    const hrMin = Math.min(...data.hrHistory)
    const hrMax = Math.max(...data.hrHistory) || hrMin + 1
    const hrRange = hrMax - hrMin || 1
    
    pdf.setDrawColor(...COLORS.accent)
    pdf.setLineWidth(0.8)
    
    for (let i = 0; i < data.hrHistory.length - 1; i++) {
      const x1 = graphX + (i / (data.hrHistory.length - 1)) * graphWidth
      const x2 = graphX + ((i + 1) / (data.hrHistory.length - 1)) * graphWidth
      const y1 = graphY + graphHeight - ((data.hrHistory[i] - hrMin) / hrRange) * (graphHeight - 10) - 5
      const y2 = graphY + graphHeight - ((data.hrHistory[i + 1] - hrMin) / hrRange) * (graphHeight - 10) - 5
      pdf.line(x1, y1, x2, y2)
    }
  }
  
  // Draw SpO2 line (blue)
  if (data.spo2History.length >= 2) {
    const spo2Min = Math.min(...data.spo2History)
    const spo2Max = Math.max(...data.spo2History) || spo2Min + 1
    const spo2Range = spo2Max - spo2Min || 1
    
    pdf.setDrawColor(...COLORS.graphBlue)
    pdf.setLineWidth(0.8)
    
    for (let i = 0; i < data.spo2History.length - 1; i++) {
      const x1 = graphX + (i / (data.spo2History.length - 1)) * graphWidth
      const x2 = graphX + ((i + 1) / (data.spo2History.length - 1)) * graphWidth
      const y1 = graphY + graphHeight - ((data.spo2History[i] - spo2Min) / spo2Range) * (graphHeight - 10) - 5
      const y2 = graphY + graphHeight - ((data.spo2History[i + 1] - spo2Min) / spo2Range) * (graphHeight - 10) - 5
      pdf.line(x1, y1, x2, y2)
    }
  }
  
  yPos = graphY + graphHeight + 8
  
  // Legend
  pdf.setFont('helvetica', 'normal')
  pdf.setFontSize(8)
  pdf.setDrawColor(...COLORS.accent)
  pdf.setLineWidth(1)
  pdf.line(graphX + 10, yPos, graphX + 25, yPos)
  pdf.setTextColor(...COLORS.textColor)
  pdf.text('Heart Rate (BPM)', graphX + 28, yPos + 1)
  
  pdf.setDrawColor(...COLORS.graphBlue)
  pdf.line(graphX + 80, yPos, graphX + 95, yPos)
  pdf.setTextColor(...COLORS.graphBlue)
  pdf.text('SpO2 (%)', graphX + 98, yPos + 1)
  
  yPos += 15

  // ==================== AI OBSERVATIONS ====================
  pdf.setFont('helvetica', 'bold')
  pdf.setFontSize(12)
  pdf.setTextColor(...COLORS.headerGray)
  pdf.setFillColor(...COLORS.bgGray)
  pdf.rect(15, yPos - 5, pageWidth - 30, 8, 'F')
  pdf.text('AI CLINICAL OBSERVATIONS', 17, yPos)
  
  yPos += 10
  
  pdf.setFont('helvetica', 'normal')
  pdf.setFontSize(10)
  pdf.setTextColor(...COLORS.textColor)
  
  const lines = pdf.splitTextToSize(data.aiMessage, pageWidth - 35)
  pdf.text(lines, 17, yPos)
  yPos += lines.length * 5 + 10

  // ==================== PRESCRIPTION SECTION ====================
  pdf.setFont('helvetica', 'bold')
  pdf.setFontSize(12)
  pdf.setTextColor(...COLORS.headerGray)
  pdf.setFillColor(...COLORS.bgGray)
  pdf.rect(15, yPos - 5, pageWidth - 30, 8, 'F')
  pdf.text('PHYSICIAN ORDERS / PRESCRIPTION', 17, yPos)
  
  yPos += 8
  
  // Draw bordered area with dotted lines
  const boxHeight = 50
  pdf.setDrawColor(150, 150, 150)
  pdf.setLineWidth(0.3)
  pdf.rect(15, yPos, pageWidth - 30, boxHeight)
  
  // Draw dotted lines for writing
  pdf.setDrawColor(...COLORS.lineGray)
  const lineSpacing = 9
  for (let i = 1; i < 6; i++) {
    const yLine = yPos + (lineSpacing * i)
    for (let x = 20; x < pageWidth - 20; x += 6) {
      pdf.line(x, yLine, Math.min(x + 3, pageWidth - 20), yLine)
    }
  }
  
  yPos += boxHeight + 10

  // ==================== FOOTER ====================
  const footerY = pdf.internal.pageSize.getHeight() - 30
  
  // Signature line
  pdf.setFont('helvetica', 'normal')
  pdf.setFontSize(10)
  pdf.setTextColor(...COLORS.textColor)
  pdf.text('__________________________', pageWidth - 15, footerY, { align: 'right' })
  pdf.setFont('helvetica', 'italic')
  pdf.setFontSize(9)
  pdf.setTextColor(...COLORS.headerGray)
  pdf.text('Attending Physician / MD', pageWidth - 15, footerY + 5, { align: 'right' })
  
  // Legal disclaimer
  pdf.setFont('helvetica', 'italic')
  pdf.setFontSize(8)
  pdf.setTextColor(...COLORS.lightGray)
  pdf.text(
    'Generated by ICU Guardian™ Clinical AI System. Not for legal diagnostic use.',
    pageWidth / 2,
    pdf.internal.pageSize.getHeight() - 10,
    { align: 'center' }
  )

  // ==================== SAVE PDF ====================
  const reportDate = new Date().toISOString().slice(0, 10).replace(/-/g, '')
  const reportTime = new Date().toTimeString().slice(0, 5).replace(':', '')
  pdf.save(`ICU_Report_${reportDate}_${reportTime}.pdf`)
}
