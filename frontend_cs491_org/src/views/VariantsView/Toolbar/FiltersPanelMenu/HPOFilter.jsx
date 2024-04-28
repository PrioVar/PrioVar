import { Grid } from '@material-ui/core'
import { HPO_OPTIONS } from 'src/constants'
import Tags from 'src/components/Tags'

function PhenotypeFilter({ hpoFilter, setHpoFilter, ...props }) {
  return (
    <Grid sx={{ display: 'flex' }}>
      {hpoFilter && setHpoFilter && (
        <Tags title={'HPO Pathogenicity'} options={HPO_OPTIONS} value={hpoFilter} onChange={setHpoFilter} fullWidth />
      )}
    </Grid>
  )
}

export default PhenotypeFilter
