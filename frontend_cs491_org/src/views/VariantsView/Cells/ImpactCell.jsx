import { Stack, Tooltip } from '@material-ui/core'
import PropTypes from 'prop-types'
import React from 'react'
import Label from 'src/components/Label'
import { groupImpacts, getImpactDescription } from 'src/utils/bio'

const renderImpactGroup = ({ values, color }) => {
  if (values.length === 0) {
    return null
  }

  const labels = values.map(({ label }) => label)
  const title = getImpactDescription(labels).join(', ')

  return (
    <Tooltip title={title} placement="top" arrow>
      <Stack direction="column" spacing={0.25}>
        <Label variant="ghost">{values.length}</Label>
      </Stack>
    </Tooltip>
  )
}

const ImpactCell = function ({ value }) {
  const { low, moderate, high, other } = groupImpacts(value)

  return (
    <Stack direction="row" spacing={0.5}>
      {renderImpactGroup({ values: high, color: 'error' })}
      {renderImpactGroup({ values: moderate, color: 'warning' })}
      {renderImpactGroup({ values: low, color: 'success' })}
      {renderImpactGroup({ values: other, color: 'default' })}
    </Stack>
  )
}

ImpactCell.propTypes = {
  value: PropTypes.any.isRequired,
}

export default ImpactCell
