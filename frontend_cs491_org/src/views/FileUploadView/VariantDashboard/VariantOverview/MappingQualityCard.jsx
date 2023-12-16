import { styled, alpha } from '@material-ui/core/styles'
import { useTheme } from '@material-ui/styles'

import { useMemo } from 'react'
import { Card, Typography, Stack, Box, CircularProgress, Paper } from '@material-ui/core'
import { ResponsiveLine } from '@nivo/line'
import { toFixedTrunc } from 'src/utils/math'
import { isNull } from 'lodash'

const RootStyle = styled(Card)(({ theme }) => ({
  // boxShadow: 'none',
  padding: theme.spacing(3),
  // color: theme.palette.primary.darker,
  // backgroundColor: theme.palette.primary.lighter,
}))

const getChartData = (data = '') => {
  if (data.length !== 0) {
    data = JSON.parse(data)
    const realData = Object.values(data.x).map((x, i) => {
      return {
        x: x,
        y: data.y[i],
      }
    })
    return [{ id: 'GC Distribution', color: 'green', data: realData }]
  }
  return null
}

const MappingQualityCard = function ({ data, title }) {
  const theme = useTheme()

  const chartData = useMemo(() => getChartData(data), [data])

  return (
    <RootStyle>
      <Stack direction="row" justifyContent="space-between">
        <Box>
          <Typography variant="subtitle1">{title}</Typography>
        </Box>
      </Stack>

      <Box sx={{ height: 280 }}>
        {chartData === null ? (
          <Stack direction="column" justifyContent="center" alignItems="center" sx={{ height: '100%' }}>
            <CircularProgress size={150} />
          </Stack>
        ) : (
          <ResponsiveLine
            data={chartData}
            margin={{ top: 50, right: 60, bottom: 50, left: 60 }}
            xScale={{ type: 'linear' }}
            yScale={{
              type: 'linear',
              min: 'auto',
              max: 'auto',
              stacked: false,
              reverse: false,
            }}
            colors={['green']}
            yFormat=" >-.2f"
            axisBottom={{
              orient: 'bottom',
              tickSize: 5,
              tickPadding: 5,
              tickRotation: 0,
              legend: 'Mapping Quality',
              legendOffset: 36,
              legendPosition: 'middle',
            }}
            axisLeft={{
              orient: 'left',
              tickSize: 5,
              tickPadding: 0,
              tickRotation: 0,
              legend: 'Number of Sequences',
              legendOffset: -50,
              legendPosition: 'middle',
              format: (v) => v.toExponential(),
            }}
            pointSize={0}
            pointColor={{ theme: 'background' }}
            pointBorderWidth={2}
            pointBorderColor={{ from: 'serieColor' }}
            pointLabelYOffset={-12}
            useMesh={true}
          />
        )}
      </Box>
    </RootStyle>
  )
}

export default MappingQualityCard
