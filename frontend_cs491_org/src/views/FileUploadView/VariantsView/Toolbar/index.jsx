import { Stack } from '@material-ui/core'
import { makeStyles } from '@material-ui/styles'

import clsx from 'clsx'
import PropTypes from 'prop-types'

import FiltersPanelMenu from './FiltersPanelMenu'

const useStyles = makeStyles((theme) => ({
  root: {
    width: '100%',
    borderRadius: theme.spacing(1, 1, 0, 0),
  },
}))

function NewToolbar({ variant = 'standard', hideButtons = false, onToolbarClose }) {
  const classes = useStyles()

  return (
    <>
      <Stack
        direction="column"
        py={1}
        px={1}
        justifyContent="space-between"
        className={clsx(classes.root, {
          [classes.standard]: variant === 'standard',
          [classes.kh1]: variant === 'kh1',
          [classes.kh2]: variant === 'kh2',
          [classes.kh3]: variant === 'kh3',
          [classes.kh4]: variant === 'kh4',
          [classes.kh5]: variant === 'kh5',
          [classes.kh6]: variant === 'kh6',
        })}
      >
        {!hideButtons && <FiltersPanelMenu onClose={onToolbarClose} />}
      </Stack>
    </>
  )
}

NewToolbar.propTypes = {
  defaultTitle: PropTypes.string.isRequired,
  readonlyTitle: PropTypes.bool.isRequired,
  variant: PropTypes.oneOf(['standard', 'kh1', 'kh2', 'kh3', 'kh4', 'kh5', 'kh6']),
  hideButtons: PropTypes.bool.isRequired,
  onToolbarClose: PropTypes.func.isRequired,
}

export default NewToolbar
