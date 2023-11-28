import { Card, Typography, Stack, Divider, CircularProgress, Tooltip, Box } from '@material-ui/core'
import React from 'react'
import Label from 'src/components/Label'
import { isHomogeneous } from 'src/utils/bio'

const GtCell = function ({ gt }) {
  return (
    <Tooltip title={gt} arrow placement="top">
      <Box>
        <Label variant="ghost" sx={{ p: 2 }}>
          <Typography variant="h4">{isHomogeneous(gt) ? 'hom' : 'het'}</Typography>
        </Label>
      </Box>
    </Tooltip>
  )
}

const renderSection = (height, title, value) => (
  <Stack direction="row" alignItems="center" justifyContent="center" spacing={3} sx={{ width: 1, height }} key={value}>
    <Stack direction="column" alignItems="center" justifyContent="center">
      <Typography variant="h6" sx={{ mb: 0.5 }} align="center">
        {value}
      </Typography>
      <Typography variant="body2" sx={{ opacity: 0.72 }}>
        {title}
      </Typography>
    </Stack>
  </Stack>
)

const LocationCard = function ({ variant, height }) {
  const { CHROM, POS, REF, ALT, GT } = variant || {}

  return (
    <Card>
      {variant === undefined ? (
        <Stack direction="column" justifyContent="center" alignItems="center" sx={{ p: 5 }}>
          <CircularProgress size={50} />
        </Stack>
      ) : (
        <Stack direction={{ xs: 'column', sm: 'row' }} divider={<Divider orientation="vertical" flexItem />}>
          {renderSection(
            height,
            'Location',
            <>
              <span>chr{CHROM}</span>
              <br />
              <span>{POS}</span>
            </>,
          )}
          {renderSection(height, 'Reference', REF)}
          {renderSection(height, 'Alternative', ALT)}
          {renderSection(height, 'Genotype', <GtCell gt={GT} />)}
        </Stack>
      )}
    </Card>
  )
}

export default LocationCard
