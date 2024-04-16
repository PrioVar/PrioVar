import { Box, Grid, Stack, Tooltip, tooltipClasses, Typography } from '@material-ui/core'
import { styled } from '@material-ui/styles'
import { orderBy } from 'lodash'
import PropTypes from 'prop-types'
import { maxBy, reduce } from 'ramda'
import React from 'react'
import ExpandOnClick from 'src/components/ExpandOnClick'
import Label from 'src/components/Label'
import { CONSEQUENCE_TO_IMPACT_MAP } from 'src/constants'

// TODO: Refactor to utils
const impactOrder = ['MODIFIER', 'LOW', 'MODERATE', 'HIGH']
const maxByImpact = maxBy((impact) => impactOrder.indexOf(impact))
const getMaxImpact = reduce(maxByImpact, 'MODIFIER')

const consequencesToImpact = (consequences) => {
  const impactsOfConsequences = consequences.map((c) => CONSEQUENCE_TO_IMPACT_MAP[c])
  return getMaxImpact(impactsOfConsequences)
}

const impactToColor = (impact) => {
  return {
    LOW: 'success',
    MODERATE: 'warning',
    HIGH: 'error',
    MODIFIER: 'default',
  }[impact]
}

const formatConsequences = (consequences) => {
  return consequences
    .join(' & ')
    .replace(/_variant/g, '')
    .replace(/_/g, '\u00a0')
}

const cleanHgvsC = (hgvsC) => {
  const [_ensemblId, rest] = hgvsC.split(':')
  return rest.length > 20 ? `${rest.slice(0, 20)}…` : rest
}

const sortEntries = (entries) => {
  const orderFn = (e) => [e.isCanonical, impactOrder.indexOf(e.impact), e.hgvsC]
  const orderDirection = ['desc', 'desc', 'asc']
  return orderBy(entries, orderFn, orderDirection)
}

const rowsToEntries = (rows) => {
  const entries = rows
    .map(([isCanonical, consequences, hgvsC]) => ({
      isCanonical,
      consequences,
      hgvsC,
      impact: consequencesToImpact(consequences),
    }))
    .map((entry) => ({ ...entry, color: impactToColor(entry.impact) }))

  return sortEntries(entries)
}

const HgvsCCell = function ({ value, variant = 'standard' }) {
  const entries = rowsToEntries(value)

  const renderedEntries = entries.map(({ consequences, hgvsC, color, isCanonical }) => (
    <Grid container item xs direction="column" justifyContent="center" alignItems="center">
      <Typography variant="caption" color="text.secondary" align="center" sx={{ fontSize: 11 }} pb={0.25}>
        {formatConsequences(consequences)}
      </Typography>
      <Label variant="ghost" color={color} sx={{ width: '100%', fontWeight: isCanonical ? 'bold' : 'unset' }}>
        {cleanHgvsC(hgvsC)}
      </Label>
    </Grid>
  ))

  const expanded = (
    <Grid
      container
      rowSpacing={0.5}
      columnSpacing={1}
      justifyContent="center"
      alignItems="center"
      sx={{ maxWidth: 450 }}
    >
      {renderedEntries}
    </Grid>
  )

  if (variant === 'expanded') {
    return expanded
  }

  return (
    <ExpandOnClick expanded={expanded}>
      {({ ref, onClick }) => (
        <Box ref={ref} onClick={renderedEntries.length > 1 ? onClick : () => {}} p={1}>
          {renderedEntries.slice(0, 1)}
        </Box>
      )}
    </ExpandOnClick>
  )
}

HgvsCCell.propTypes = {
  value: PropTypes.array.isRequired,
  variant: PropTypes.oneOf(['standard', 'expanded']),
}

export default HgvsCCell
