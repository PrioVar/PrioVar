import { Box, LinearProgress, linearProgressClasses, Stack, Tooltip, Typography } from '@material-ui/core'
import { styled } from '@material-ui/styles'
import { darken, alpha } from '@material-ui/core/styles'
import PropTypes from 'prop-types'
import React from 'react'

const GRADIENT_COLORS = [
  '#ffffcc',
  '#ffeda0',
  '#fed976',
  '#feb24c',
  '#fd8d3c',
  '#fc4e2a',
  '#e31a1c',
  '#bd0026',
  '#800026',
]

const BG_GRADIENT_COLORS = GRADIENT_COLORS.map((color) => darken(color, 0.5)).map((color) => alpha(color, 0.5))

const ChromaLinearProgress = styled(LinearProgress)({
  height: 13,
  borderRadius: 6.5,
  opacity: 0.6,
  [`&.${linearProgressClasses.colorPrimary}`]: {
    background: `linear-gradient(0.25turn, ${BG_GRADIENT_COLORS.join(', ')})`,
  },
  [`& .${linearProgressClasses.bar}`]: {
    transform: 'unset !important',
    clipPath: (props) => `inset(0 ${Math.trunc(100 - props.value)}% 0 0 round 6.5px)`,
    background: `linear-gradient(0.25turn, ${GRADIENT_COLORS.join(', ')})`,
  },
})

const LibraPathogenicityCell = function ({ value = null }) {
  if (value === null) {
    return (
      <Tooltip title="Pathogenicity Unknown" arrow placement="top">
        <Box sx={{ position: 'relative' }}>
          <ChromaLinearProgress variant="determinate" value={0} sx={{ flexGrow: 1 }} />
          <Typography
            variant="caption"
            sx={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)' }}
          >
            N/A
          </Typography>
        </Box>
      </Tooltip>
    )
  }

  return (
    <Tooltip title="Overall Pathogenicity" arrow placement="top">
      <Stack direction="row" alignItems="center" justifyContent="center" spacing={0.5} px={0.25}>
        <ChromaLinearProgress variant="determinate" value={Math.trunc(value * 100)} sx={{ flexGrow: 1 }} />
        <Typography variant="caption">{Math.trunc(value * 100)}%</Typography>
      </Stack>
    </Tooltip>
  )
}

LibraPathogenicityCell.propTypes = {
  value: PropTypes.number,
}

export default LibraPathogenicityCell
