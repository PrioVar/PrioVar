import { styled, alpha } from '@material-ui/core/styles'
import { useTheme } from '@material-ui/styles'

import { useMemo } from 'react'
import { Card, Typography, Stack, Box, CircularProgress, Paper } from '@material-ui/core'
import { ResponsiveBar } from '@nivo/bar'
import { toFixedTrunc } from 'src/utils/math'

const RootStyle = styled(Card)(({ theme }) => ({
  // boxShadow: 'none',
  padding: theme.spacing(3),
  /*  color: theme.palette.primary.darker,
  backgroundColor: theme.palette.primary.lighter, */
}))

const formatValue = (value) => {
  const f = toFixedTrunc(value, 2)
  return f === 'NaN' ? null : f
}

const getChartData = (file = {}) => {
  return [
    {
      value: 'A',
      depth: 158,
      depthColor: 'green',
    },
    {
      value: 'G',
      depth: 87,
      depthColor: 'green',
    },
    {
      value: 'C',
      depth: 156,
      depthColor: 'green',
    },
    {
      value: 'T',
      depth: 125,
      depthColor: 'green',
    },
  ]
}

const BarChartCard = function ({ file, title }) {
  const theme = useTheme()

  const chartData = useMemo(() => getChartData(file), [file])

  return (
    <RootStyle>
      <Stack direction="row" justifyContent="space-between">
        <Box>
          <Typography variant="h5">{title}</Typography>
        </Box>
      </Stack>

      <Box sx={{ height: 280 }}>
        {file === undefined ? (
          <Stack direction="column" justifyContent="center" alignItems="center" sx={{ height: '100%' }}>
            <CircularProgress size={150} />
          </Stack>
        ) : (
          <ResponsiveBar
            data={chartData}
            keys={['depth']}
            indexBy="value"
            margin={{ top: 50, right: 130, bottom: 50, left: 60 }}
            padding={0.3}
            valueScale={{ type: 'linear' }}
            indexScale={{ type: 'band', round: true }}
            colors={{ scheme: 'dark2' }}
            axisTop={null}
            axisRight={null}
            axisBottom={{
              tickSize: 5,
              tickPadding: 5,
              tickRotation: 0,
              legend: 'value',
              legendPosition: 'middle',
              legendOffset: 32,
            }}
            axisLeft={{
              tickSize: 5,
              tickPadding: 5,
              tickRotation: 0,
              legend: 'depth',
              legendPosition: 'middle',
              legendOffset: -40,
            }}
            labelSkipWidth={12}
            labelSkipHeight={12}
            labelTextColor={{
              from: 'color',
              modifiers: [['darker', 1.6]],
            }}
            legends={[
              {
                dataFrom: 'keys',
                anchor: 'bottom-right',
                direction: 'column',
                justify: false,
                translateX: 120,
                translateY: 0,
                itemsSpacing: 2,
                itemWidth: 100,
                itemHeight: 20,
                itemDirection: 'left-to-right',
                itemOpacity: 0.85,
                symbolSize: 20,
                effects: [
                  {
                    on: 'hover',
                    style: {
                      itemOpacity: 1,
                    },
                  },
                ],
              },
            ]}
            role="application"
            ariaLabel="Nivo bar chart demo"
            barAriaLabel={function (e) {
              return e.id + ': ' + e.formattedValue + ' in value: ' + e.indexValue
            }}
          />
        )}
      </Box>
    </RootStyle>
  )
}

export default BarChartCard
