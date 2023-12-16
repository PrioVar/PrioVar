import React from 'react'
import Label from 'src/components/Label'

const JobStateStatus = function ({ status = 'PENDING', isAnalysis = false }) {
  const { color, label } = {
    VERIFYING: { color: 'warning', label: 'Verifying' },
    PENDING: { color: 'secondary', label: isAnalysis ? 'In queue' : 'Ready to process' },
    RUNNING: { color: 'warning', label: 'Processing' },
    DONE: { color: 'success', label: 'Analyzed' },
    FAILED: { color: 'error', label: 'Failed' },
    UNKNOWN: { color: 'error', label: 'Unknown' },
    WAITING: { color: 'warning', label: 'Phenotype Data Required' },
    ANNO_PENDING: { color: 'secondary', label: 'Annotation In Queue' },
    ANNO_RUNNING: { color: 'warning', label: 'Annotation Processing' },
    ANNO_FAILED: { color: 'error', label: 'Annotation Failed' },
  }[status]

  return (
    <Label color={color} variant="ghost" sx={{ px: 1, py: 1.75 }}>
      {label}
    </Label>
  )
}

export default JobStateStatus
