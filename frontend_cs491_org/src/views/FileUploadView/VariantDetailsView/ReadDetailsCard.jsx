import { Card, Typography, Stack, Divider, CircularProgress } from '@material-ui/core'
import { toFixedTrunc } from 'src/utils/math'

const renderSection = (height, title, value) => (
  <Stack direction="row" alignItems="center" justifyContent="center" spacing={3} sx={{ width: 1, height }} key={value}>
    <Stack direction="column" alignItems="center" justifyContent="center">
      <Typography variant="h6" sx={{ mb: 0.5 }}>
        {value}
      </Typography>
      <Typography variant="body2" sx={{ opacity: 0.72 }}>
        {title}
      </Typography>
    </Stack>
  </Stack>
)

const ReadDetailsCard = function ({ variant, height }) {
  const { Read_Details = {} } = variant || {}
  const { AD = [], DP } = Read_Details || {}

  return (
    <Card>
      {variant === undefined ? (
        <Stack direction="column" justifyContent="center" alignItems="center" sx={{ p: 5 }}>
          <CircularProgress size={50} />
        </Stack>
      ) : (
        <Stack direction={{ xs: 'column', sm: 'row' }} divider={<Divider orientation="vertical" flexItem />}>
          {renderSection(height, 'Read Depth', DP)}
          {renderSection(height, 'Allelic Depth', AD.join(' | '))}
          {renderSection(
            height,
            'Allelic Balance',
            AD.slice(1)
              .map((ad) => ad / DP)
              .map((value) => toFixedTrunc(value, 2)),
          )}
        </Stack>
      )}
    </Card>
  )
}

export default ReadDetailsCard
