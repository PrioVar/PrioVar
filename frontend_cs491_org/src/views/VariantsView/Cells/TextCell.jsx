import { Tooltip, Typography } from '@material-ui/core'
import PropTypes from 'prop-types'
import { isNil } from 'ramda'
import React, { useState } from 'react'
import { toFixedTrunc } from 'src/utils/math'

const formatFloat = (value) => {
  const float = toFixedTrunc(value, 2)
  return float === 'NaN' ? '' : float
}

const formatInteger = (value) => {
  const integer = parseInt(`${value}`, 10)
  return `${integer}` === 'NaN' ? '' : integer
}

const formatString = (value) => {
  if (isNil(value)) {
    return ''
  }
  return `${value}`
}

const formatValue = (value, dataType) => {
  const formatSingularValue = (singularValue) => {
    switch (dataType) {
      case 'float':
        return formatFloat(singularValue)
      case 'integer':
        return formatInteger(singularValue)
      case 'string':
      default:
        return formatString(singularValue)
    }
  }

  if (Array.isArray(value)) {
    return value
      .map(formatSingularValue)
      .filter((x) => x !== '')
      .join(', ')
  }
  return formatSingularValue(value)
}

const elementHasTextOverflow = (element) => {
  return element && element.offsetWidth && element.scrollWidth && element.offsetWidth < element.scrollWidth
}

const TextCell = function ({ value, dataType }) {
  // using useState instead of useRef to trigger re-render
  const [textRef, setTextRef] = useState()

  const title = formatValue(value, dataType)

  const isNumeric = dataType === 'integer' || dataType === 'float'

  return (
    <Tooltip title={elementHasTextOverflow(textRef) ? title : ''} placement="top" interactive arrow leaveDelay={50}>
      <Typography variant="body2" noWrap ref={setTextRef} align={isNumeric ? 'right' : 'left'}>
        {title}
      </Typography>
    </Tooltip>
  )
}

TextCell.propTypes = {
  value: PropTypes.any.isRequired,
  dataType: PropTypes.oneOf(['string', 'integer', 'float']).isRequired,
}

export default TextCell
