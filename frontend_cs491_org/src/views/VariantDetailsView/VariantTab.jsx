import { Box, Divider, Grid, Link, Stack, Tooltip, Typography } from '@material-ui/core'
import React from 'react'
import { useParams } from 'react-router-dom'

const renderDatabases = (databases) => {
  if (databases.length === 0) {
    return <span>–</span>
  }

  const dbSnpIds = databases.filter((id) => id.startsWith('rs'))
  const cosmicIds = databases.filter((id) => id.startsWith('COSV'))

  return (
    <Stack direction="row" spacing={0.5}>
      {dbSnpIds.map((id) => (
        <Link target="_blank" href={`https://www.ncbi.nlm.nih.gov/snp/${encodeURIComponent(id)}`}>
          {id}
        </Link>
      ))}
      {cosmicIds.map((id) => (
        <Link target="_blank" href={`https://cancer.sanger.ac.uk/cosmic/search?q=${encodeURIComponent(id)}`}>
          {id}
        </Link>
      ))}
    </Stack>
  )
}

const renderColumn = (title, content, tooltip = null) => {
  const titleEl = <Typography variant="overline">{title}</Typography>

  return (
    <Grid item xs px={4} py={1}>
      {tooltip ? (
        <Tooltip title={tooltip} placement="right-end">
          {titleEl}
        </Tooltip>
      ) : (
        titleEl
      )}
      <Divider sx={{ width: '100%', borderColor: 'rgba(145, 158, 171, 0.6)' }} />
      <Typography variant="body2" noWrap>
        {content}
      </Typography>
    </Grid>
  )
}

const VariantTab = function ({ data }) {
  const { fileId, sampleName, chrom, pos } = useParams()

  const geneIdsSet = new Set(data.Transcripts.map((t) => t.Gene_Symbol))
  const geneIds = Array.from(geneIdsSet).sort()

  return (
    <>
      <Grid container>
        {renderColumn(
          'SAMPLE',
          <Link target="_blank" href={`/priovar/variants/${fileId}/${sampleName}`}>
            {sampleName}
          </Link>,
        )}
        {renderColumn('CHROMOSOME', `chr${chrom}`)}
        {renderColumn('POSITION', pos)}
        {renderColumn('REF', data.REF, 'Reference')}
        {renderColumn('ALT', data.ALT, 'Alternative')}
        {renderColumn('DP', data.DP, 'Read Depth')}
        {renderColumn('Databases', renderDatabases(data.Databases))}
        {renderColumn(
          'Gene ID',
          geneIds.map((geneId) => (
            <Link
              key={geneId}
              target="_blank"
              href={`https://pubchem.ncbi.nlm.nih.gov/gene/${geneId}/human`}
              sx={{ pr: 0.5 }}
            >
              {geneId}
            </Link>
          )),
        )}
      </Grid>
      <Box p={1} />
      <Box display="flex" flexDirection="column">
        <Typography variant="subtitle1">Additional links</Typography>
        <Box pl={2} pt={1}>
          <Link
            target="_blank"
            href={`https://genome.ucsc.edu/cgi-bin/hgTracks?db=hg38&position=chr${chrom}:${pos}-${pos}`}
          >
            UCSC Genome Browser
          </Link>
        </Box>
      </Box>
    </>
  )
}

export default VariantTab
