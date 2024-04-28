import { Stack } from '@material-ui/core'
import PropTypes from 'prop-types'
import React from 'react'

const splitIntoChunks = (chunkSize, array) => {
  const result = []

  for (let i = 0; i < array.length; i += chunkSize) {
    const chunk = array.slice(i, i + chunkSize)
    result.push(chunk)
  }

  return result
}

function ResponsiveGrid({ children, columns, rowProps = {}, rowSpacing, columnSpacing, ...rest }) {
  const childrenArray = React.Children.toArray(children).filter(Boolean)
  const rows = splitIntoChunks(columns, childrenArray)

  return (
    <Stack direction="column" spacing={columnSpacing} {...rest}>
      {rows.map((row, rowIndex) => (
        <Stack key={rowIndex} direction="row" spacing={rowSpacing} {...rowProps}>
          {row}
        </Stack>
      ))}
    </Stack>
  )
}

ResponsiveGrid.propTypes = {
  children: PropTypes.node.isRequired,
  columns: PropTypes.number.isRequired,
  rowProps: PropTypes.any,
  rowSpacing: PropTypes.number,
  columnSpacing: PropTypes.number,
}

export default ResponsiveGrid
