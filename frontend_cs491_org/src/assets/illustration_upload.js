// material
import { useTheme } from '@material-ui/core/styles'
import { Box } from '@material-ui/core'

// ----------------------------------------------------------------------

export default function UploadIllustration({ ...other }) {
  const theme = useTheme()

  return (
    <Box {...other} sx={{ paddingLeft: theme.spacing(1) }}>
        <img alt="illustration_upload" src="/static/new_images/upload_file.png" x="0" y="0" height="150px" width="150px"/>
    </Box>
  )
}
