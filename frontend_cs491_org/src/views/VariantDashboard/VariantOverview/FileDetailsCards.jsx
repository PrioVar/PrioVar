import { styled, alpha } from '@material-ui/core/styles'
import { useTheme } from '@material-ui/styles'
import { Card, Typography, Stack, Box, Grid } from '@material-ui/core'

// icons
import { Waves, DonutLarge, ChromeReaderMode, Numbers } from '@mui/icons-material'

import { abbreviateNumber } from 'src/utils/math'
import { useMemo } from 'react'

const RootStyle = styled(Card)(({ theme }) => ({
  padding: theme.spacing(3),
  // color: theme.palette.primary.darker,
  // backgroundColor: theme.palette.primary.lighter,
}))

const OneCard = ({ title, value, icon }) => (
  <RootStyle>
    <Stack direction="row" alignItems={'center'} spacing={2}>
      {icon}
      <Stack direction="row" justifyContent="space-between">
        <Box>
          <Typography variant="h7">{title}</Typography>
          <Typography variant="h4">{value}</Typography>
        </Box>
      </Stack>
    </Stack>
  </RootStyle>
)

const FileDetailsCards = function ({ avgDepth, targetCov, noOfReads, sx }) {
  const theme = useTheme()
  const avgDepthValue = useMemo(() => {
    return Math.round(avgDepth * 100) / 100
  }, [avgDepth])
  const targetCovValue = useMemo(() => {
    return Math.round(targetCov * 100)
  }, [targetCov])
  const noOfReadsValue = useMemo(() => {
    return abbreviateNumber(noOfReads)
  }, [noOfReads])

  return (
    <Grid container spacing={2} sx={sx}>
      <Grid item xs={12} md={4} lg={4}>
        <OneCard
          title="Average Depth"
          value={avgDepthValue}
          icon={<Waves sx={{ fontSize: 40, color: theme.palette.primary }} />}
        />
      </Grid>
      <Grid item xs={12} md={4} lg={4}>
        <OneCard
          title="Target Coverage"
          value={`${targetCovValue}%`}
          icon={<DonutLarge sx={{ fontSize: 40, color: theme.palette.primary }} />}
        />
      </Grid>
      <Grid item xs={12} md={4} lg={4}>
        <OneCard
          title="Number of Reads in Target"
          value={noOfReadsValue}
          icon={<ChromeReaderMode sx={{ fontSize: 40, color: theme.palette.primary }} />}
        />
      </Grid>
    </Grid>
  )
}

export default FileDetailsCards
