import { Autocomplete, TextField, Button, Grid, Stack, CircularProgress, Card } from '@material-ui/core'
import { useMemo, useState } from 'react'
import CovAccordion from './GeneCovAccordion'
import { getCoverage } from 'src/api/fastq'
import { GENE_CHOICES } from 'src/constants'
import { ResponsiveBar } from '@nivo/bar'
import { DataSaverOff } from '@material-ui/icons'

const Loading = function () {
  return (
    <Stack direction="row" justifyContent="center" marginTop={10}>
      <CircularProgress size="10vh" />
    </Stack>
  )
}

const GeneCov = ({ fileId }) => {
  const [gene, setGene] = useState('')
  const [coverage, setCoverage] = useState(0)
  const [loading, setLoading] = useState(false)
  const [chartData, setChartData] = useState([])
  const [avgCov, setAvgCov] = useState(0)

  const filtered = useMemo(() => {
    return GENE_CHOICES?.filter((o) => ![gene].includes(o))
  }, [gene])

  const handleClick = async () => {
    setLoading(true)
    const data = await getCoverage(fileId, gene, coverage)
    setChartData(data)
    //console.log(data)
    const biggest = data.reduce((prev, current) =>
      prev.transcript_result.length > current.transcript_result.length ? prev : current,
    )
    const avg =
      biggest.transcript_result.reduce((prev, current) => prev + current.coverage, 0) / biggest.transcript_result.length
    setAvgCov(avg)
    setLoading(false)
  }

  return (
    <Grid container spacing={2} direction="column">
      <Grid
        container
        item
        xs={12}
        spacing={2}
        sx={{ marginTop: '5px', justifyContent: 'center', alignItems: 'center' }}
      >
        <Grid item xs={4}>
          <Autocomplete
            options={filtered}
            renderInput={(params) => <TextField {...params} label="Gene" variant="outlined" />}
            value={gene}
            getOptionLabel={(option) => option || ''}
            onChange={(_e, newGene) => {
              setGene(newGene)
            }}
          />
        </Grid>
        <Grid item xs={2}>
          <TextField
            label="Target Coverage"
            value={coverage}
            onChange={(e) => setCoverage(e.target.value)}
            placeholder="coverage"
          ></TextField>
        </Grid>
        <Grid item xs={0}>
          <Button variant="contained" onClick={handleClick}>
            Submit
          </Button>
        </Grid>
      </Grid>
      {loading && <Loading />}
      {avgCov !== 0 && !loading && (
        <Card
          sx={{
            paddingY: '30px',
            marginX: 'auto',
            marginTop: '20px',
            paddingX: '20px',
            justifyContent: 'center',
            alignItems: 'center',
            display: 'flex',
            gridGap: '10px',
          }}
        >
          <DataSaverOff />
          <h3>Average Coverage: {avgCov.toFixed(2)}%</h3>
        </Card>
      )}
      {chartData.length !== 0 && !loading && (
        <Grid container spacing={2} sx={{ marginTop: '5px', justifyContent: 'center', alignItems: 'center' }}>
          {chartData?.map((item) => (
            <Grid item xs={12} key={item.transcript_name}>
              <CovAccordion
                title={`Transcript: ${item.transcript_name} | Chromosome: ${item.transcript_chr}`}
                height="280px"
                defaultExpand={false}
              >
                <ResponsiveBar
                  data={item.transcript_result}
                  keys={['coverage']}
                  indexBy="exome_index"
                  margin={{ top: 10, right: 60, bottom: 50, left: 60 }}
                  axisBottom={{
                    tickSize: 5,
                    tickPadding: 5,
                    tickRotation: 0,
                    legend: 'Exomes',
                    legendPosition: 'middle',
                    legendOffset: 32,
                  }}
                  axisLeft={{
                    tickSize: 5,
                    tickPadding: 5,
                    tickRotation: 0,
                    legend: 'Coverage Percentage Above ' + coverage,
                    legendPosition: 'middle',
                    legendOffset: -40,
                  }}
                  colors={['#ccebc5']}
                  isInteractive
                  tooltip={({ data }) => {
                    return (
                      <div
                        style={{
                          color: 'black',
                          backgroundColor: 'white',
                          border: '1px solid rgba(0, 0, 0, 0.2)',
                          borderRadius: '5px',
                          boxShadow: '1px 1px 1px 1px rgba(0, 0, 0, 0.2)',
                          padding: '5px',
                          opacity: '0.9',
                        }}
                      >
                        <strong>Location:</strong> {data.location}
                        <br />
                        <strong>Coverage Above {coverage} Bases:</strong> {data.coverage}%
                      </div>
                    )
                  }}
                />
              </CovAccordion>
            </Grid>
          ))}
        </Grid>
      )}
    </Grid>
  )
}

export default GeneCov
