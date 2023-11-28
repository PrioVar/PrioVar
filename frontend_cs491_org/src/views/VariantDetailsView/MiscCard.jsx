import { Box, Card, CircularProgress, Divider, Grid, Stack, Tooltip, Typography } from '@material-ui/core'
import React from 'react'
import Label from 'src/components/Label'
import { getImpactDescription, groupImpacts } from 'src/utils/bio'
import GeneSymbolsCell from 'src/views/VariantsView/Cells/GeneSymbolsCell'

const renderImpactGroup = ({ values, color }) => {
  if (values.length === 0) {
    return null
  }

  const labels = values.map(({ label }) => label)
  const descriptions = getImpactDescription(labels)

  return descriptions.map((d) => (
    <Grid item key={d}>
      <Label color={color} variant="ghost">
        {d}
      </Label>
    </Grid>
  ))
}

const ImpactBlock = function ({ impacts }) {
  const { low, moderate, high, other } = groupImpacts(impacts)

  return (
    <Grid container spacing={0.5} px={2} justifyContent="center" alignItems="center">
      {renderImpactGroup({ values: high, color: 'error' })}
      {renderImpactGroup({ values: moderate, color: 'warning' })}
      {renderImpactGroup({ values: low, color: 'success' })}
      {renderImpactGroup({ values: other, color: 'info' })}
    </Grid>
  )
}

const renderSection = (height, title, value) => (
  <Stack direction="row" alignItems="center" justifyContent="center" spacing={3} sx={{ width: 1, height }}>
    <Stack direction="column" alignItems="center" justifyContent="center">
      <Stack direction="row" spacing={1} sx={{ mb: 1 }}>
        {value}
      </Stack>
      <Typography variant="body2" sx={{ opacity: 0.72 }}>
        {title}
      </Typography>
    </Stack>
  </Stack>
)

const MiscCard2 = function ({ variant }) {
  const { Gene_Symbols: geneSymbols = [], Consequences: consequences = [] } = variant || {}

  return (
    <Card>
      {variant === undefined ? (
        <Stack direction="column" justifyContent="center" alignItems="center" sx={{ p: 5 }}>
          <CircularProgress size={50} />
        </Stack>
      ) : (
        <Stack direction={{ xs: 'column', sm: 'row' }} divider={<Divider orientation="vertical" flexItem />}>
          {renderSection('Gene Symbols', <GeneSymbolsCell symbols={geneSymbols} />)}
          {renderSection('Impact', <ImpactBlock impacts={consequences} />)}
        </Stack>
      )}
    </Card>
  )
}

const GeneSmybolsCard = function ({ variant, height }) {
  const { Gene_Symbols: geneSymbols = [] } = variant || {}

  return (
    <Card>
      {variant === undefined ? (
        <Stack direction="column" justifyContent="center" alignItems="center" sx={{ p: 5 }}>
          <CircularProgress size={50} />
        </Stack>
      ) : (
        <Stack direction={{ xs: 'column', sm: 'row' }} divider={<Divider orientation="vertical" flexItem />}>
          {renderSection(height, 'Gene Symbols', <GeneSymbolsCell symbols={geneSymbols} limit={4} />)}
        </Stack>
      )}
    </Card>
  )
}

const ImpactsCard = function ({ variant, height }) {
  const { Consequences: consequences = [] } = variant || {}

  return (
    <Card>
      {variant === undefined ? (
        <Stack direction="column" justifyContent="center" alignItems="center" sx={{ p: 5 }}>
          <CircularProgress size={50} />
        </Stack>
      ) : (
        <Stack direction={{ xs: 'column', sm: 'row' }} divider={<Divider orientation="vertical" flexItem />}>
          {renderSection(height, 'Impact', <ImpactBlock impacts={consequences} />)}
        </Stack>
      )}
    </Card>
  )
}

export { GeneSmybolsCard, ImpactsCard }
