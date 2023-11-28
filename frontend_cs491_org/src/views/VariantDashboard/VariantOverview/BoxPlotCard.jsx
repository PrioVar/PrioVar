import { styled, alpha } from '@material-ui/core/styles'
import { useTheme } from '@material-ui/styles'

import { useMemo } from 'react'
import { Card, Typography, Stack, Box, CircularProgress, Paper } from '@material-ui/core'
import { ResponsiveBar } from '@nivo/bar'

const RootStyle = styled(Card)(({ theme }) => ({
  // boxShadow: 'none',
  padding: theme.spacing(3),
  /*  color: theme.palette.primary.darker,
  backgroundColor: theme.palette.primary.lighter, */
}))

const CUSTOM_BAR_PROPS = {
  whiskerWidth: 20,
  whiskerHeight: 2,
  lineWidth: 1,
  boxWidth: 100,
  whiskerStart: 179,
  whiskerColor: '#566450',
  lineColor: '#566450',
  boxColor: 'lightgreen',
}

const CustomBarComponent = ({ bar }) => {
  const totalWidth = bar.width + 2 * bar.x
  if (bar.data.id === 'downWhisker') {
    const whiskerX = (totalWidth - CUSTOM_BAR_PROPS['whiskerWidth']) / 2
    return (
      <rect
        x={whiskerX}
        y={CUSTOM_BAR_PROPS['whiskerStart']}
        width={CUSTOM_BAR_PROPS['whiskerWidth']}
        height={CUSTOM_BAR_PROPS['whiskerHeight']}
        fill={CUSTOM_BAR_PROPS['whiskerColor']}
      />
    )
  } else if (bar.data.id === 'firstQuartile') {
    const lineX = (totalWidth - CUSTOM_BAR_PROPS['lineWidth']) / 2
    return (
      <rect
        x={lineX}
        y={bar.y}
        width={CUSTOM_BAR_PROPS['lineWidth']}
        height={bar.height}
        fill={CUSTOM_BAR_PROPS['lineColor']}
      />
    )
  } else if (bar.data.id === 'median') {
    const boxX = (totalWidth - CUSTOM_BAR_PROPS['boxWidth']) / 2
    return (
      <rect
        x={boxX}
        y={bar.y}
        width={CUSTOM_BAR_PROPS['boxWidth']}
        height={bar.height}
        fill={CUSTOM_BAR_PROPS['boxColor']}
        stroke="black"
      />
    )
  } else if (bar.data.id === 'thirdQuartile') {
    const boxX = (totalWidth - CUSTOM_BAR_PROPS['boxWidth']) / 2
    return (
      <>
        <rect
          x={boxX}
          y={bar.y}
          width={CUSTOM_BAR_PROPS['boxWidth']}
          height={bar.height}
          fill={CUSTOM_BAR_PROPS['boxColor']}
          stroke="black"
        />
      </>
    )
  } else if (bar.data.id === 'upWhisker') {
    const whiskerX = (totalWidth - CUSTOM_BAR_PROPS['whiskerWidth']) / 2
    const lineX = (totalWidth - CUSTOM_BAR_PROPS['lineWidth']) / 2
    return (
      <>
        <rect
          x={lineX}
          y={bar.y}
          width={CUSTOM_BAR_PROPS['lineWidth']}
          height={bar.height}
          fill={CUSTOM_BAR_PROPS['lineColor']}
        />
        <rect
          x={whiskerX}
          y={bar.y}
          width={CUSTOM_BAR_PROPS['whiskerWidth']}
          height={CUSTOM_BAR_PROPS['whiskerHeight']}
          fill={CUSTOM_BAR_PROPS['whiskerColor']}
        />
      </>
    )
  }

  return <rect x={10} y={180} width={20} height={20} />
}

const chartData = (data) => {
  if (data) {
    return [
      {
        value: 'TEST',
        downWhisker: data.min,
        downWhiskerColor: 'hsl(213, 70%, 50%)',
        firstQuartile: data.first - data.min,
        firstQuartileColor: 'hsl(213, 70%, 50%)',
        median: data.median - data.first,
        medianColor: 'hsl(213, 70%, 50%)',
        thirdQuartile: data.last - data.median,
        thirdQuartileColor: 'hsl(213, 70%, 50%)',
        upWhisker: data.max - data.last,
        upWhiskerColor: 'hsl(213, 70%, 50%)',
      },
    ]
  }
  return null
}

const BoxPlotCard = function ({ data, title }) {
  const cData = useMemo(() => chartData(data), [data])
  return (
    <RootStyle>
      <Stack direction="row" justifyContent="space-between">
        <Box>
          <Typography variant="subtitle1">{title}</Typography>
        </Box>
      </Stack>

      <Box sx={{ height: 280, display: 'flex', justifyContent: 'center' }}>
        {cData === null ? (
          <Stack direction="column" justifyContent="center" alignItems="center" sx={{ height: '100%' }}>
            <CircularProgress size={150} />
          </Stack>
        ) : (
          <ResponsiveBar
            data={cData}
            margin={{ top: 50, right: 60, bottom: 50, left: 60 }}
            indexBy="value"
            keys={['downWhisker', 'firstQuartile', 'median', 'thirdQuartile', 'upWhisker']}
            isInteractive
            axisLeft={{
              legend: 'Read Depth',
              legendPosition: 'middle',
              legendOffset: -40,
            }}
            axisBottom={null}
            valueScale={{ type: 'linear' }}
            indexScale={{ type: 'band', round: true }}
            minValue={0}
            maxValue={data?.last + 100}
            barComponent={CustomBarComponent}
          />
        )}
      </Box>
    </RootStyle>
  )
}

export default BoxPlotCard
