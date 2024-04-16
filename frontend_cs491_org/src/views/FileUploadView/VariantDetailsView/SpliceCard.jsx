import { useTheme } from '@material-ui/styles'
import { alpha } from '@material-ui/core/styles'
import { useEffect, useMemo, useState } from 'react'
import {
  Card,
  Typography,
  Stack,
  Box,
  CircularProgress,
  Paper,
  CardHeader,
  TextField,
  Tooltip,
} from '@material-ui/core'
import { ResponsiveBar } from '@nivo/bar'
import { sortBy } from 'ramda'

const getChartData = (variant = {}) => {
  const { Splice_AI: spliceAI = {} } = variant

  const entries = Object.entries(spliceAI).map(([geneSymbol, spliceScores]) => {
    const [acceptorGain, acceptorLoss, donorGain, donorLoss] = spliceScores

    const value = [
      { name: 'Acceptor Gain', value: acceptorGain },
      { name: 'Acceptor Loss', value: acceptorLoss },
      { name: 'Donor Gain', value: donorGain },
      { name: 'Donor Loss', value: donorLoss },
      { name: '', value: '' },
    ]

    return { geneSymbol, value }
  })

  return sortBy((entry) => entry.geneSymbol, entries)
}

const SpliceCard = function ({ variant }) {
  const theme = useTheme()

  const chartData = useMemo(() => getChartData(variant), [variant])
  const [selectedIndex, setSelectedIndex] = useState(0)

  return (
    <Card>
      <CardHeader
        title="Splice AI"
        action={
          <Tooltip title="Gene Symbol" placement="top" arrow>
            <TextField
              select
              fullWidth
              SelectProps={{ native: true }}
              value={selectedIndex}
              onChange={(e) => setSelectedIndex(e.target.value)}
              sx={{
                '& fieldset': { border: '0 !important' },
                '& select': { pl: 1, py: 0.5, pr: '24px !important', typography: 'subtitle2' },
                '& .MuiOutlinedInput-root': { borderRadius: 0.75, bgcolor: 'background.neutral' },
                '& .MuiNativeSelect-icon': { top: 4, right: 0, width: 20, height: 20 },
              }}
            >
              {chartData.map(({ geneSymbol }, index) => (
                <option key={geneSymbol} value={index}>
                  {geneSymbol}
                </option>
              ))}
            </TextField>
          </Tooltip>
        }
      />

      <Box sx={{ height: 300 }} p={1}>
        {variant === undefined ? (
          <Stack direction="column" justifyContent="center" alignItems="center" sx={{ height: '100%' }}>
            <CircularProgress size={150} />
          </Stack>
        ) : (
          <ResponsiveBar
            data={chartData?.[selectedIndex]?.value ?? []}
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
            axisLeft={{
              legend: 'Score',
              legendPosition: 'middle',
              legendOffset: -50,
            }}
            minValue={0}
            maxValue={1}
            markers={[
              {
                axis: 'y',
                value: 0.2,
                lineStyle: { stroke: alpha(theme.palette.primary.dark, 0.75), strokeWidth: 2 },
                legend: 'High recall',
                legendOrientation: 'horizontal',
              },
              {
                axis: 'y',
                value: 0.5,
                lineStyle: { stroke: alpha(theme.palette.warning.main, 0.75), strokeWidth: 2 },
                legend: 'Recommended',
                legendOrientation: 'horizontal',
              },
              {
                axis: 'y',
                value: 0.8,
                lineStyle: { stroke: alpha(theme.palette.error.main, 0.75), strokeWidth: 2 },
                legend: 'Precise',
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
    </Card>
  )
}

export default SpliceCard
