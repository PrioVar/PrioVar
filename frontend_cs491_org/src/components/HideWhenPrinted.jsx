import React from 'react'
import { styled } from '@material-ui/core/styles'

const HideWhenPrinted = styled('div')({
  '@media print': {
    display: 'none',
  },
})

export default HideWhenPrinted
