import { /*Badge,*/ Box, /*Grid,*/ Stack, Tooltip, Typography } from '@material-ui/core'
import { countBy } from 'lodash'
import PropTypes from 'prop-types'
import React from 'react'
import Label from 'src/components/Label'
import { CLIN_VAR_SEVERITY } from 'src/constants'

const groupClinVar = (values) => {
  const low = values.filter((v) => CLIN_VAR_SEVERITY.LOW.includes(v))
  const moderate = values.filter((v) => CLIN_VAR_SEVERITY.MODERATE.includes(v))
  const high = values.filter((v) => CLIN_VAR_SEVERITY.HIGH.includes(v))
  const other = values.filter((v) => !low.includes(v) && !moderate.includes(v) && !high.includes(v))

  return { low, moderate, high, other }
}

const getTitle = (labels) => {
  const countMap = countBy(labels)

  return Object.entries(countMap)
    .map(([label, count]) => `${count} ✖ ${label}`)
    .join(', ')
}

const renderImpactGroup = ({ values, color }) => {
  if (values.length === 0) {
    return null
  }

  const labels = values.map((v) => v.replace(/_/g, ' '))
  const title = getTitle(labels)

  return (
    <Tooltip title={title} placement="top" leaveDelay={50} arrow interactive>
      <Box>
        <Label color={color} variant="ghost">
          {values.length}
        </Label>
      </Box>
    </Tooltip>
  )
}

const ClinVarCell = function ({ value }) {
  /* FIXME: fix clinvar data on the server side */
  value = value?.[0]?.split(',') ?? null

  if (!value || !value.length) {
    return null
  }

  const { low, moderate, high, other } = groupClinVar(value)

  return (
    <Stack direction="column" alignItems="center" spacing={0.25}>
      <Typography variant="caption" color="text.secondary" sx={{ width: 45, fontSize: 11, textAlign: 'center' }}>
        ClinVar
      </Typography>
      <Stack direction="row" spacing={0.5}>
        {renderImpactGroup({ values: high, color: 'error' })}
        {renderImpactGroup({ values: moderate, color: 'warning' })}
        {renderImpactGroup({ values: low, color: 'success' })}
        {renderImpactGroup({ values: other, color: 'default' })}
      </Stack>
    </Stack>
  )
}

ClinVarCell.propTypes = {
  value: PropTypes.any.isRequired,
}

export default ClinVarCell
