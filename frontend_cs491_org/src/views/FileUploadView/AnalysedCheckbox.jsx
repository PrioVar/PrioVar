import { Box, Checkbox, Typography } from '@material-ui/core'
import { fDateTime } from 'src/utils/formatTime'

function AnalysedCheckbox({ checked, onChange, details, user, ...rest }) {
  return (
    <Box display="flex" flexDirection="column" alignItems="center">
      <Checkbox checked={checked} onChange={onChange} {...rest} />
      {details.date && checked && (
        <Typography variant="caption" sx={{ fontSize: 9 }}>
          {fDateTime(details.date)}
        </Typography>
      )}
      {details.person && checked && (
        <Typography variant="caption" sx={{ fontSize: 9 }}>
          by {details.person}
        </Typography>
      )}
    </Box>
  )
}

export default AnalysedCheckbox
