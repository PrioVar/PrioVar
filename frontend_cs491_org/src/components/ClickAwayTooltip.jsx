import { Box, ClickAwayListener, Tooltip } from '@material-ui/core'
import { useState } from 'react'

function ClickAwayTooltip({ children, ...rest }) {
  const [open, setOpen] = useState(false)

  const handleTooltipClose = () => {
    setOpen(false)
  }

  const handleTooltipOpen = () => {
    setOpen(true)
  }

  return (
    <ClickAwayListener onClickAway={handleTooltipClose}>
      <Tooltip
        onClose={handleTooltipClose}
        open={open}
        disableFocusListener
        disableHoverListener
        disableTouchListener
        {...rest}
      >
        <Box onClick={handleTooltipOpen}>{children}</Box>
      </Tooltip>
    </ClickAwayListener>
  )
}

export default ClickAwayTooltip
