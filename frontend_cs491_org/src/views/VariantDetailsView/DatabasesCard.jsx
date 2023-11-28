import { Card, Typography, Stack, Divider, CircularProgress, Link } from '@material-ui/core'
import React from 'react'
import { toFixedTrunc } from 'src/utils/math'

const renderSection = (height, title, values) => (
  <Stack direction="row" alignItems="center" justifyContent="center" spacing={3} sx={{ width: 1, height }}>
    <Stack direction="column" alignItems="center" justifyContent="center">
      <Stack direction="column" spacing={1}>
        {values.map(({ value, url }) => (
          <Link target="_blank" rel="noopener" href={url} sx={{ color: 'inherit' }}>
            <Typography variant="h6" sx={{ mb: 0.5 }}>
              {value}
            </Typography>
          </Link>
        ))}
      </Stack>
      <Typography variant="body2" sx={{ opacity: 0.72 }}>
        {title}
      </Typography>
    </Stack>
  </Stack>
)

const DatabasesCard = function ({ variant, height }) {
  const { Databases: ids = [], CHROM, POS } = variant || {}

  const dbSnpIds = ids.filter((id) => id.startsWith('rs'))
  const cosmicIds = ids.filter((id) => id.startsWith('COSV'))

  return (
    <Card>
      {variant === undefined ? (
        <Stack direction="column" justifyContent="center" alignItems="center" sx={{ p: 5 }}>
          <CircularProgress size={50} />
        </Stack>
      ) : (
        <Stack direction={{ xs: 'column', sm: 'row' }} divider={<Divider orientation="vertical" flexItem />}>
          {renderSection(height, 'VARSOME', [
            {
              value: `${CHROM}-${POS}`,
              url: `https://varsome.com/position/hg38/chr${CHROM}%20${POS}`,
            },
          ])}
          {renderSection(
            height,
            'DBSNP',
            dbSnpIds.length === 0
              ? [{ value: '?', url: '#' }]
              : dbSnpIds.map((id) => ({
                  value: id,
                  url: `https://www.ncbi.nlm.nih.gov/snp/${encodeURIComponent(id)}`,
                })),
          )}
          {renderSection(
            height,
            'COSMIC',
            cosmicIds.length === 0
              ? [{ value: '?', url: '#' }]
              : cosmicIds.map((id) => ({
                  value: id,
                  url: `https://cancer.sanger.ac.uk/cosmic/search?q=${encodeURIComponent(id)}`,
                })),
          )}
        </Stack>
      )}
    </Card>
  )
}

export default DatabasesCard
