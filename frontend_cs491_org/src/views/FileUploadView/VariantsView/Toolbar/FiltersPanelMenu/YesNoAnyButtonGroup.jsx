import React from 'react'
import PropTypes from 'prop-types'
import { Box, ToggleButton, ToggleButtonGroup, Typography } from '@material-ui/core'

function YesNoAnyButtonGroup({ title, value, onChange, copy = { any: 'Any', yes: 'Yes', no: 'No' } }) {
  return (
    <Box display="flex" alignItems="center" justifyContent="space-between">
      <Typography gutterBottom>{title}</Typography>
      <ToggleButtonGroup value={value} exclusive size="small">
        <ToggleButton value={'ANY'} onClick={() => onChange('ANY')}>
          {copy.any}
        </ToggleButton>
        <ToggleButton value={'YES'} onClick={() => onChange('YES')}>
          {copy.yes}
        </ToggleButton>
        <ToggleButton value={'NO'} onClick={() => onChange('NO')}>
          {copy.no}
        </ToggleButton>
      </ToggleButtonGroup>
    </Box>
  )
}
YesNoAnyButtonGroup.propTypes = {
  value: PropTypes.string.isRequired,
  onChange: PropTypes.func.isRequired,
  copy: PropTypes.shape({
    any: PropTypes.string.isRequired,
    yes: PropTypes.string.isRequired,
    no: PropTypes.string.isRequired,
  }),
}

export default YesNoAnyButtonGroup
