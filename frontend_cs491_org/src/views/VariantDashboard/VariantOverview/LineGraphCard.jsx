import { styled, alpha } from '@material-ui/core/styles'
import { useTheme } from '@material-ui/styles'

import { useMemo } from 'react'
import { Card, Typography, Stack, Box, CircularProgress, Paper } from '@material-ui/core'
import { ResponsiveLine } from '@nivo/line'
import { toFixedTrunc } from 'src/utils/math'

const RootStyle = styled(Card)(({ theme }) => ({
  // boxShadow: 'none',
  padding: theme.spacing(3),
  // color: theme.palette.primary.darker,
  // backgroundColor: theme.palette.primary.lighter,
}))

const getChartData = (data = []) => {
  return [
    {
      id: 'Bases Above',
      color: 'hsl(213, 70%, 50%)',
      data: [
        {
          x: '1',
          y: data[0] * 100,
        },
        {
          x: '5',
          y: data[1] * 100,
        },
        {
          x: '10',
          y: data[2] * 100,
        },
        {
          x: '20',
          y: data[3] * 100,
        },
        {
          x: '30',
          y: data[4] * 100,
        },
        {
          x: '40',
          y: data[5] * 100,
        },
        {
          x: '50',
          y: data[6] * 100,
        },
        {
          x: '75',
          y: data[7] * 100,
        },
        {
          x: '100',
          y: data[8] * 100,
        },
      ],
    },
  ]
}

const LineGraphCard = function ({ data, title }) {
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
        {data === undefined ? (
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
            yFormat=" <-.2f"
            axisBottom={{
              orient: 'bottom',
              tickSize: 5,
              tickPadding: 5,
              tickRotation: 0,
              legend: 'Threshold',
              legendOffset: 36,
              legendPosition: 'middle',
            }}
            axisLeft={{
              orient: 'left',
              tickSize: 5,
              tickPadding: 5,
              tickRotation: 0,
              legend: '% of Bases Above',
              legendOffset: -40,
              legendPosition: 'middle',
            }}
            pointSize={10}
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

export default LineGraphCard
