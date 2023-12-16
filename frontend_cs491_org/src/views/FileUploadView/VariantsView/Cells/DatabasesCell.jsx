import { Link, Stack, Typography } from '@material-ui/core'
import PropTypes from 'prop-types'
import React from 'react'
import Label from 'src/components/Label'
import ResponsiveGrid from 'src/components/ResponsiveGrid'
import ClinVarCell from 'src/views/VariantsView/Cells/ClinVarCell' // `https://www.ncbi.nlm.nih.gov/snp/${encodeURIComponent(id)}`

// `https://www.ncbi.nlm.nih.gov/snp/${encodeURIComponent(id)}`

const renderTitle = (title) => {
  return (
    <Typography variant="caption" color="text.secondary" sx={{ fontSize: 11, textAlign: 'center' }}>
      {title}
    </Typography>
  )
}

const renderLabel = (label, url) => {
  return (
    <Label variant="ghost">
      <Link target="_blank" rel="noopener" href={url} sx={{ color: 'inherit' }}>
        {label}
      </Link>
    </Label>
  )
}

const DatabasesCell = function ({ ids, CHROM, POS }) {
  const dbSnpIds = ids.filter((id) => id.startsWith('rs'))
  const cosmicIds = ids.filter((id) => id.startsWith('COSV'))

  return (
    <ResponsiveGrid columns={3} rowSpacing={0.5} columnSpacing={0.5}>
      <Stack direction="column" spacing={0.25} key={POS}>
        {renderTitle('Varsome')}
        {renderLabel(`${CHROM}-${POS}`, `https://varsome.com/position/hg38/chr${CHROM}%20${POS}`)}
      </Stack>
      {dbSnpIds.slice(0, 1).map((id) => (
        <Stack direction="column" spacing={0.25} key={id}>
          {renderTitle('dbSNP')}
          {renderLabel(id, `https://www.ncbi.nlm.nih.gov/snp/${encodeURIComponent(id)}`)}
        </Stack>
      ))}
      {cosmicIds.slice(0, 1).map((id) => (
        <Stack direction="column" spacing={0.25} key={id}>
          {renderTitle('COSMIC')}
          {renderLabel(id, `https://cancer.sanger.ac.uk/cosmic/search?q=${encodeURIComponent(id)}`)}
        </Stack>
      ))}
    </ResponsiveGrid>
  )
}

DatabasesCell.propTypes = {
  ids: PropTypes.any.isRequired,
  CHROM: PropTypes.string.isRequired,
  POS: PropTypes.string.isRequired,
}

export default DatabasesCell
