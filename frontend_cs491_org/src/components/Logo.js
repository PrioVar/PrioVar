// material
import { Box } from '@material-ui/core'
import { makeStyles } from '@material-ui/styles'
import { mt } from 'date-fns/locale'

import PropTypes from 'prop-types'

// ----------------------------------------------------------------------

Logo.propTypes = {
  sx: PropTypes.object,
}

const useStyles = makeStyles({
  img: {
    borderRadius: 3,
    width: '100%',
  },
})

export default function Logo({ sx, size = 'small' }) {
  const classes = useStyles()

  switch (size) {
    case 'large':
      return (
        <Box sx={{ width: { xs: 54, md: 256 }, mt:5, ...sx }}>
          <img src="/static/new_images/priovar_logo.png" className={classes.img} />
        </Box>
      )
    case 'small':
    default:
      if (window.devicePixelRatio > 1) {
        return (
          <Box sx={{ width: 54, ...sx }}>
            <img src="/static/new_images/priovar_logo.png" className={classes.img} />
          </Box>
        )
      }
      return (
        <Box sx={{ width: 54, ...sx }}>
          <img src="/static/new_images/priovar_logo.png" className={classes.img} />
        </Box>
      )
  }
}
