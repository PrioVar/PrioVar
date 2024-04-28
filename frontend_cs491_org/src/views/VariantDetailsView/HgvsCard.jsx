import { /*Box,*/ Card, CircularProgress, Divider, Grid, Stack, /*Tooltip,*/ Typography } from '@material-ui/core'
import React from 'react'
//import Label from 'src/components/Label'
//import { getImpactDescription, groupImpacts } from 'src/utils/bio'
//import GeneSymbolsCell from 'src/views/VariantsView/Cells/GeneSymbolsCell'
import HgvsCCell from 'src/views/VariantsView/Cells/HgvsCCell'



const HgvsCBlock = function ({ hgvsC }) {
  return (
    <Grid container spacing={0.5} px={2} justifyContent="center" alignItems="center">
      <HgvsCCell value={hgvsC} variant="expanded" />
    </Grid>
  )
}

const renderSection = (title, value) => (
  <Stack direction="column" alignItems="center" justifyContent="center" p={1}>
    <Stack direction="row" spacing={1} sx={{ mb: 1 }}>
      {value}
    </Stack>
    <Typography variant="body2" sx={{ opacity: 0.72 }}>
      {title}
    </Typography>
  </Stack>
)

const HgvsCard = function ({ variant }) {
  const { HGVSc = [], HGVSp = [] } = variant || {}

  return (
    <Card sx={{ height: '100%' }}>
      <Stack direction="column" alignItems="center" justifyContent="space-evenly" sx={{ height: '100%' }}>
        {variant === undefined ? (
          <CircularProgress size={50} />
        ) : (
          [
            renderSection('HGVSc', <HgvsCBlock hgvsC={HGVSc} />),
            <Divider orientation="horizontal" flexItem sx={{ pt: 1 }} />,
            renderSection('HGVSp', <HgvsCBlock hgvsC={HGVSp} />),
          ]
        )}
      </Stack>
    </Card>
  )
}

export default HgvsCard

/*
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
*/