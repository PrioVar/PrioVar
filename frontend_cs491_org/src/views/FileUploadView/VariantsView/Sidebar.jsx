import { Box, Drawer } from '@material-ui/core'
import { makeStyles } from '@material-ui/styles'
import PropTypes from 'prop-types'
import React from 'react'

const SIDEBAR_WIDTH = 550

const useStyles = makeStyles({
  backdrop: {
    opacity: (props) => (props.hideBackdrop ? '0 !important' : '1'),
  },
  sidebar: {
    width: SIDEBAR_WIDTH,
  },
})

function Sidebar({ isOpen, onClose, hideBackdrop, anchor = 'right', children }) {
  const classes = useStyles({ hideBackdrop })

  return (
    <Drawer
      anchor={anchor}
      open={isOpen}
      onClose={onClose}
      ModalProps={{
        BackdropProps: {
          className: classes.backdrop,
        },
      }}
    >
      <Box className={classes.sidebar}>{children}</Box>
    </Drawer>
  )
}

Sidebar.propTypes = {
  isOpen: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  children: PropTypes.node.isRequired,
  hideBackdrop: PropTypes.bool.isRequired,
  anchor: PropTypes.oneOf(['left', 'right']),
}

Sidebar.defaultProps = {
  anchor: 'right',
}

export default Sidebar
