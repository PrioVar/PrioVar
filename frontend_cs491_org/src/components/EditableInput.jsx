import { TextField } from '@material-ui/core'
import React, { useState } from 'react'

const FONT = '16px "Exo 2"'

const getTextWidth = (text, font) => {
  const { canvas } = getTextWidth

  const context = canvas.getContext('2d')
  context.font = font

  const { width } = context.measureText(text)
  return width
}
getTextWidth.canvas = document.createElement('canvas')

// TextField in MUI doesn't autosiez the width
function EditableInput({ readOnly, value, ...rest }) {
  const [isFocused, setIsFocused] = useState(false)

  return (
    <TextField
      InputProps={{
        disableUnderline: true,
        readOnly,
        sx: {
          width: getTextWidth(value, FONT) + 30,
          '& > fieldset': {
            border: !readOnly && isFocused ? undefined : 'unset',
          },
        },
      }}
      value={value}
      onFocus={() => setIsFocused(true)}
      onBlur={() => setIsFocused(false)}
      size="small"
      margin="dense"
      variant="outlined"
      {...rest}
    />
  )
}

export default EditableInput
