// material-ui
import { Box, Card, Grid } from '@material-ui/core'
// api
import { getPlots } from 'src/api/file/'
// components
import FileDetailsCards from './VariantOverview/FileDetailsCards'
import LineGraphCard from './VariantOverview/LineGraphCard'
import BarChartCard from './VariantOverview/BarChartCard'
import BoxPlotCard from './VariantOverview/BoxPlotCard'
import GCDistributionCard from './VariantOverview/GCDistributionCard'
import SequenceLengthCard from './VariantOverview/SequenceLengthCard'
import MappingQualityCard from './VariantOverview/MappingQualityCard'
import { useEffect, useState } from 'react'

const VariantOverview = ({ file, fileId, sampleName }) => {
  const [plots, setPlots] = useState(null)
  useEffect(() => {
    if (file.vcf_id) return
    getPlots(fileId).then((res) => {
      setPlots(res)
    })
  }, [fileId, file.vcf_id])

  const dashboardObject = file?.annotations?.list.dashboard
  return (
    <Grid container spacing={2}>
      <FileDetailsCards
        avgDepth={dashboardObject?.avg_depth}
        targetCov={dashboardObject?.b1}
        noOfReads={dashboardObject?.total_reads}
        sx={{ marginTop: '5px' }}
      />
      <Grid item xl={4} md={4} xs={12}>
        <BoxPlotCard
          data={{
            first: dashboardObject?.first_quartile,
            last: dashboardObject?.last_quartile,
            median: dashboardObject?.median,
            max: dashboardObject?.max_depth,
            min: dashboardObject?.min_depth,
          }}
          title={'Read Depth Box Plot'}
        />
      </Grid>
      <Grid item xl={4} md={4} xs={12}>
        <LineGraphCard
          data={[
            dashboardObject?.b1,
            dashboardObject?.b5,
            dashboardObject?.b10,
            dashboardObject?.b20,
            dashboardObject?.b30,
            dashboardObject?.b40,
            dashboardObject?.b50,
            dashboardObject?.b75,
            dashboardObject?.b100,
          ]}
          title={'Bases Above'}
        />
      </Grid>
      <Grid item xl={4} md={4} xs={12}>
        <GCDistributionCard data={plots?.gcd} title={'GC Distribution Over All Sequences'} />
      </Grid>
      <Grid item xl={6} md={6} xs={12}>
        <SequenceLengthCard data={plots?.seqlen} title={'Distribution of Sequence Length Over All Sequences'} />
      </Grid>
      <Grid item xl={6} md={6} xs={12}>
        <MappingQualityCard data={plots?.mapq} title={'Distribution of Mapping Quality Over All Sequences'} />
      </Grid>
    </Grid>
  )
}

export default VariantOverview
