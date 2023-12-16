import { styled, alpha } from '@material-ui/core/styles'
import { useTheme } from '@material-ui/styles'

import { useMemo } from 'react'
import { Card, Typography, Stack, Box, CircularProgress, Paper } from '@material-ui/core'
import { ResponsiveBar } from '@nivo/bar'
import { toFixedTrunc } from 'src/utils/math'

const RootStyle = styled(Card)(({ theme }) => ({
  // boxShadow: 'none',
  padding: theme.spacing(3),
  // color: theme.palette.primary.darker,
  // backgroundColor: theme.palette.primary.lighter,
}))

const formatValue = (value) => {
  const f = toFixedTrunc(value, 2)
  return f === 'NaN' ? null : f
}

const getChartData = (variant = {}) => {
  const { Pathogenicity: pathogenicity = {} } = variant

  return [
    { name: 'SIFT', value: formatValue(pathogenicity.SIFT) },
    { name: 'Polyphen', value: formatValue(pathogenicity.Polyphen) },
    { name: 'DANN', value: formatValue(pathogenicity.DANN) },
    { name: 'MetalR', value: formatValue(pathogenicity.MetalR) },
    { name: 'CADD', value: formatValue(pathogenicity.CADD) },
    { name: 'REVEL', value: formatValue(pathogenicity.REVEL) },
    { name: 'Lidya', value: formatValue(pathogenicity.LibraP) },
    { name: '', value: '' },
  ]
}

const PathogenicityCard = function ({ variant }) {
  const theme = useTheme()

  const chartData = useMemo(() => getChartData(variant), [variant])

  return (
    <RootStyle>
      <Stack direction="row" justifyContent="space-between">
        <Box>
          <Typography variant="h5">Pathogenicity</Typography>
          {/* <Typography variant="h3">{fCurrency(TOTAL)}</Typography> */}
        </Box>
        {/* <Stack direction="row" alignItems="center" justifyContent="flex-end" sx={{ mb: 0.6 }}>
          <IconWrapperStyle>
            <LocalHospitalIcon fontSize="medium" />
          </IconWrapperStyle>
        </Stack> */}

        {/*        <Box>
          <Stack direction="row" alignItems="center" justifyContent="flex-end" sx={{ mb: 0.6 }}>
            <Icon width={20} height={20} icon={PERCENT >= 0 ? trendingUpFill : trendingDownFill} />
            <Typography variant="subtitle2" component="span" sx={{ ml: 0.5 }}>
              {PERCENT > 0 && '+'}
              {fPercent(PERCENT)}
            </Typography>
          </Stack>
          <Typography variant="body2" component="span" sx={{ opacity: 0.72 }}>
            &nbsp;than last month
          </Typography>
        </Box> */}
      </Stack>

      <Box sx={{ height: 280 }}>
        {variant === undefined ? (
          <Stack direction="column" justifyContent="center" alignItems="center" sx={{ height: '100%' }}>
            <CircularProgress size={150} />
          </Stack>
        ) : (
          <ResponsiveBar
            data={chartData}
            keys={['value']}
            indexBy="name"
            margin={{ top: 10, right: 0, bottom: 30, left: 50 }}
            padding={0.7}
            colors={alpha(theme.palette.primary.main, 0.7)}
            animate
            borderRadius={5}
            borderWidth={2}
            borderColor={theme.palette.primary.dark}
            enableLabel={false}
            axisLeft={{
              legend: 'Score',
              legendPosition: 'middle',
              legendOffset: -40,
            }}
            minValue={0}
            maxValue={1}
            markers={[
              {
                axis: 'y',
                value: 0.3,
                lineStyle: { stroke: alpha(theme.palette.primary.dark, 0.75), strokeWidth: 2 },
                legend: 'Benign',
                legendOrientation: 'horizontal',
              },
              {
                axis: 'y',
                value: 0.7,
                lineStyle: { stroke: alpha(theme.palette.error.main, 0.75), strokeWidth: 2 },
                legend: 'Pathogenic',
                legendOrientation: 'horizontal',
              },
            ]}
            tooltip={({ indexValue, formattedValue }) => (
              <Paper elevation={5} sx={{ p: 1 }}>
                <Typography>
                  {indexValue}: {formattedValue}
                </Typography>
              </Paper>
            )}
          />
        )}
      </Box>
    </RootStyle>
  )
}

export default PathogenicityCard
