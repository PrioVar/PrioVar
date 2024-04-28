import { Stack, /*Tooltip,*/ Typography } from '@material-ui/core'
import React from 'react'
import ClickAwayTooltip from 'src/components/ClickAwayTooltip'
import Label from 'src/components/Label'
import { toFixedTrunc } from 'src/utils/math'

const formatPositiveValue = (value) => {
  const decimalValueArray = `${value}`.split('.')
  if (decimalValueArray.length === 1) {
    return `${value}`
  }

  const decimalValue = parseFloat(`0.${decimalValueArray[1]}`, 10)

  if (decimalValue < 0.001 && value <= 1) {
    return `${toFixedTrunc(value * 10000, 1)}‱`
  }
  if (decimalValue < 0.01) {
    return toFixedTrunc(value, 3)
  }
  return toFixedTrunc(value, 2)
}

const formatValue = (value) => {
  if (value >= 0) {
    return formatPositiveValue(value)
  }
  return `-${formatPositiveValue(-value)}`
}

export var LabelledFloat = function ({ label, value, color }) {
  if (value === undefined || value === null) {
    return null
  }

  const formattedValue = formatValue(value)
  // Use double equal sign to compare
  // This is one of the few times where we really want this behaviour
  // e.g compare '0.50' with 0.5
  // eslint-disable-next-line eqeqeq
  const tooltip = value != formattedValue ? value : ''

  return (
    <Stack direction="column" px={0.25} spacing={0.25} justifyContent="center" alignItems="center">
      <Typography variant="caption" color="text.secondary" align="center" sx={{ fontSize: 11 }}>
        {label}
      </Typography>
      <ClickAwayTooltip title={tooltip} arrow>
        <Stack justifyContent="center" alignItems="center">
          <Label color={color} variant="ghost" sx={{ width: 45 }}>
            {formattedValue}
          </Label>
        </Stack>
      </ClickAwayTooltip>
    </Stack>
  )
}
