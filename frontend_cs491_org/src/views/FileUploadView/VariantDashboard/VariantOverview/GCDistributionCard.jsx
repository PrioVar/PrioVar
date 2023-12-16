import { styled, alpha } from '@material-ui/core/styles'
import { useTheme } from '@material-ui/styles'

import { useMemo } from 'react'
import { Card, Typography, Stack, Box, CircularProgress, Paper } from '@material-ui/core'
import { ResponsiveLine } from '@nivo/line'

const RootStyle = styled(Card)(({ theme }) => ({
  // boxShadow: 'none',
  padding: theme.spacing(3),
  // color: theme.palette.primary.darker,
  // backgroundColor: theme.palette.primary.lighter,
}))

const getZScore = (x, mean, stdev) => {
  const lhs = 1 / Math.sqrt(2 * Math.PI * stdev * stdev)
  const rhs = Math.pow(Math.E, 0 - Math.pow(x - mean, 2) / (2 * stdev * stdev))
  return lhs * rhs
}

const calculateTheoreticalGC = (gcDistribution, length) => {
  let gcContent = []
  for (let i = 0; i < length; i++) {
    gcContent.push(parseFloat(gcDistribution[i]))
  }
  let max = 0
  let totalCount = 0
  const xCategories = []
  for (let i = 0; i < length; i++) {
    xCategories.push(0)
  }

  let firstMode = 0
  let modeCount = 0

  for (let i = 0; i < length; i++) {
    xCategories[i] = i
    totalCount += parseFloat(gcContent[i])
    if (gcContent[i] > modeCount) {
      modeCount = gcContent[i]
      firstMode = i
    }
    if (gcContent[i] > max) max = gcContent[i]
  }
  // The mode might not be a very good measure of the centre
  // of the distribution either due to duplicated vales or
  // several very similar values next to each other.  We therefore
  // average over adjacent points which stay above 95% of the modal
  // value

  let mode = 0
  let modeDuplicates = 0

  let fellOffTop = true

  for (let i = firstMode; i < gcContent.length; i++) {
    if (gcContent[i] > gcContent[firstMode] - gcContent[firstMode] / 10) {
      mode += i
      modeDuplicates++
    } else {
      fellOffTop = false
      break
    }
  }

  let fellOffBottom = true

  for (let i = firstMode - 1; i >= 0; i--) {
    if (gcContent[i] > gcContent[firstMode] - gcContent[firstMode] / 10) {
      mode += i
      modeDuplicates++
    } else {
      fellOffBottom = false
      break
    }
  }

  if (fellOffBottom || fellOffTop) {
    // If the distribution is so skewed that 95% of the mode
    // is off the 0-100% scale then we keep the mode as the
    // centre of the model
    mode = firstMode
  } else {
    mode /= modeDuplicates
  }

  // We can now work out a theoretical distribution
  let stdev = 0

  for (let i = 0; i < gcContent.length; i++) {
    stdev += Math.pow(i - mode, 2) * gcContent[i]
  }

  stdev /= totalCount - 1

  stdev = Math.sqrt(stdev)

  const theoreticalDistribution = []

  for (let i = 0; i < length; i++) {
    const probability = getZScore(i, mode, stdev)
    theoreticalDistribution.push(Math.round(probability * totalCount * 100) / 100)

    if (theoreticalDistribution[i] > max) {
      max = theoreticalDistribution[i]
    }
  }
  return theoreticalDistribution
}

const getChartData = (data = '') => {
  if (data.length !== 0) {
    data = JSON.parse(data)
    const realData = Object.values(data.x).map((x, i) => {
      return {
        x: x,
        y: data.y[i],
      }
    })
    const theory = calculateTheoreticalGC(Object.values(data.y), 101)
    const theoryData = theory.map((x, i) => {
      return {
        x: data.x[i],
        y: theory[i],
      }
    })
    return [
      { id: 'GC Distribution', color: 'green', data: realData },
      { id: 'Theoretical', color: 'red', data: theoryData },
    ]
  }
  return null
}

const GCDistributionCard = function ({ data, title }) {
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
            margin={{ top: 50, right: 10, bottom: 50, left: 60 }}
            xScale={{ type: 'linear' }}
            yScale={{
              type: 'linear',
              min: '0',
              max: 'auto',
              stacked: false,
              reverse: false,
            }}
            colors={['green', 'red']}
            axisBottom={{
              legend: 'Mean GC Content',
              legendOffset: 36,
              legendPosition: 'middle',
            }}
            axisLeft={{
              legend: 'Number of Sequences',
              legendOffset: -50,
              legendPosition: 'middle',
              format: (value) => `${value.toExponential()}`,
            }}
            pointSize={10}
            pointColor={{ theme: 'background' }}
            enableSlices="x"
            sliceTooltip={({ slice }) => {
              return (
                <div
                  sx={{ p: 1 }}
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
                  <strong>Mean GC Content: </strong>
                  {slice.points[0].data.xFormatted}
                  <br />
                  <strong>Number of Sequences: </strong>
                  {slice.points[0].data.yFormatted}
                  <br />
                  <strong>Theoretical Distribution: </strong>
                  {slice.points[1].data.yFormatted}
                </div>
              )
            }}
            legends={[
              {
                anchor: 'top-right',
                direction: 'column',
                justify: false,
                translateX: -20,
                translateY: 0,
                itemsSpacing: 0,
                itemDirection: 'left-to-right',
                itemWidth: 80,
                itemHeight: 20,
                itemOpacity: 1,
                symbolSize: 12,
                symbolShape: 'circle',
                symbolBorderColor: 'rgba(0, 0, 0, .5)',
                effects: [
                  {
                    on: 'hover',
                    style: {
                      itemBackground: 'rgba(0, 0, 0, .03)',
                      itemOpacity: 1,
                    },
                  },
                ],
              },
            ]}
          />
        )}
      </Box>
    </RootStyle>
  )
}

export default GCDistributionCard
