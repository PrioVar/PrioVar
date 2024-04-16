import { styled, alpha } from '@material-ui/core/styles'
import { useTheme } from '@material-ui/styles'

import { useMemo } from 'react'
import {
  Card,
  Typography,
  Stack,
  Box,
  CircularProgress,
  Paper,
  CardHeader,
  Tooltip,
  TextField,
} from '@material-ui/core'
import { ResponsiveBar } from '@nivo/bar'
import { fShortenNumber } from 'src/utils/formatNumber'
import FrequencyCell from '../VariantsView/Cells/FrequencyCell'

const formatValue = (value) => fShortenNumber(value * 100)

const getChartData = (variant = {}) => {
  const { Frequency: frequency = {} } = variant

  return [
    { name: 'Local', value: formatValue(frequency.AF) },
    { name: 'GnomAD', value: formatValue(frequency.GnomAd_Af) },
    { name: '1KG', value: formatValue(frequency.One_Kg_Af) },
    { name: 'ExAC', value: formatValue(frequency.ExAC_AF) },
    { name: 'TurkishVariome', value: formatValue(frequency.TV_AF) },
  ]
}

const FrequencyCard2 = function ({ variant, height }) {
  const theme = useTheme()

  const chartData = useMemo(() => getChartData(variant), [variant])

  return (
    <Card>
      <CardHeader title="Population Frequency" />

      <Box sx={{ height }} p={1}>
        {variant === undefined ? (
          <Stack direction="column" justifyContent="center" alignItems="center" sx={{ height: '100%' }}>
            <CircularProgress size={150} />
          </Stack>
        ) : (
          <ResponsiveBar
            data={chartData}
            keys={['value']}
            indexBy="name"
            margin={{ top: 10, right: 0, bottom: 30, left: 60 }}
            padding={0.7}
            colors={alpha(theme.palette.primary.main, 0.7)}
            animate
            borderRadius={5}
            borderWidth={3}
            borderColor={theme.palette.primary.dark}
            enableLabel={false}
            valueScale={{ type: 'symlog' }}
            valueFormat={(value) => `${value}%`}
            axisLeft={{
              legend: 'Frequency (%)',
              legendPosition: 'middle',
              legendOffset: -50,
            }}
            minValue={0}
            maxValue={100}
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
    </Card>
  )
}

const FrequencyCard = function ({ variant, height }) {
  const { Frequency: frequency = {} } = variant || {}

  return (
    <Card>
      <CardHeader title="Population Frequency" />
      <Box sx={{ height }} p={1}>
        {variant === undefined ? (
          <Stack direction="column" justifyContent="center" alignItems="center" sx={{ height: '100%' }}>
            <CircularProgress size={150} />
          </Stack>
        ) : (
          <FrequencyCell
            libraAf={frequency.AF}
            gnomAdAf={frequency.GnomAd_Af}
            oneKgAf={frequency.One_Kg_Af}
            exAcAf={frequency.ExAC_AF}
            tvAf={frequency.TV_AF}
          />
        )}
      </Box>
    </Card>
  )
}

export default FrequencyCard
