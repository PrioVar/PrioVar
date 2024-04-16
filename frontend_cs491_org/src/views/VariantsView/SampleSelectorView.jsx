import React from 'react'
import SampleSelector from 'src/components/Medical/SampleSelector'
import { Typography } from '@material-ui/core'

function SampleSelectorView() {
  return (
    <div>
      <Typography gutterBottom variant="subtitle2" sx={{ color: 'text.secondary' }}>
        Available Samples
      </Typography>
      <SampleSelector autoRedirect />
    </div>
  )
}

export default SampleSelectorView
