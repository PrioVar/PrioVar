import { Card, CardHeader, Typography, Link, /*Stack,*/ CardContent, Grid, Divider } from '@material-ui/core'
import { useParams } from 'react-router-dom'

function MatchmakingCard({ matchmaking, height }) {
  const { chrom, pos } = useParams()

  const {
    own_vcf_files: ownVcfFiles = [],
    other_vcf_files_count: otherVcfFilesCount = 0,
    organisation_name: organisationName = '',
  } = matchmaking || {}

  return (
    <Card sx={{ height }}>
      <CardHeader title="Variant Occurrence" />
      <CardContent>
        <Grid container direction="row" justifyContent="space-between">
          <Grid container item xs={5} direction="column" alignItems="center">
            <Typography variant="h6">In {organisationName}</Typography>
            {ownVcfFiles.map((vcfFile) => (
              <Link href={`/priovar/variants/${vcfFile.id}/${vcfFile.sample}/${chrom}/${pos}`} target="_blank">
                <Typography variant="body2">{vcfFile.sample}</Typography>
              </Link>
            ))}
          </Grid>
          <Grid container item xs={2} justifyContent="center" my={-1}>
            <Divider orientation="vertical" flexItem />
          </Grid>
          <Grid container item xs={5} direction="column" alignItems="center">
            <Typography variant="h6">In Community</Typography>
            <Typography variant="body2">{otherVcfFilesCount} samples</Typography>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  )
}

export default MatchmakingCard
