import { Popover } from '@material-ui/core'
import React, { useRef, useState } from 'react'

function ExpandOnClick({ children, expanded, ...rest }) {
  const anchorElRef = useRef()
  const [popoverIsOpen, setPopoverIsOpen] = useState(false)

  const handlePopoverOpen = () => {
    setPopoverIsOpen(true)
  }

  const handlePopoverClose = () => {
    setPopoverIsOpen(false)
  }

  return (
    <>
      {children({ ref: anchorElRef, onClick: handlePopoverOpen })}
      <Popover
        open={popoverIsOpen}
        anchorEl={anchorElRef.current}
        anchorOrigin={{
          vertical: 'center',
          horizontal: 'left',
        }}
        transformOrigin={{
          vertical: 'center',
          horizontal: 'left',
        }}
        onClose={handlePopoverClose}
        disableRestoreFocus
        keepMounted
        {...rest}
      >
        {expanded}
      </Popover>
    </>
  )
}

export default ExpandOnClick
