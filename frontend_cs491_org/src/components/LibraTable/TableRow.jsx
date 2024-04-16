import { Skeleton, TableCell, TableRow } from '@material-ui/core'
import { alpha } from '@material-ui/core/styles'
import { makeStyles } from '@material-ui/styles'
import clsx from 'clsx'
import PropTypes from 'prop-types'
import React from 'react'

const useStyles = makeStyles((theme) => ({
  root: {
    '&:nth-child(even)': {
      background: theme.palette.background.neutral,
    },
    '&:hover': {
      background: alpha(theme.palette.action.hover, 0.3),
    },
  },
  cell: {
    overflow: 'hidden',
    padding: theme.spacing(0.75),
  },
}))

function LibraTableRow({ columns, row, isLoading, iconColumns }) {
  const classes = useStyles()

  return (
    <TableRow className={classes.root} tabIndex={-1}>
      {iconColumns.map((iconColumn, index) => {
        const values = iconColumn.indexes.map((i) => row[i])
        values.push(row[28])

        return (
          <TableCell key={index} padding="checkbox" className="collapsed">
            {iconColumn.renderCell(values)}
          </TableCell>
        )
      })}
      {columns.map((column) => {
        const values = column.indexes.map((i) => row[i])
        const renderedCell = column.renderCell(values)

        return (
          <TableCell
            key={column.label}
            className={clsx(classes.cell, {
              collapsed: column.collapse,
            })}
            sx={{ maxWidth: column.width || 'unset' }}
          >
            {isLoading ? <Skeleton>{renderedCell}</Skeleton> : renderedCell}
          </TableCell>
        )
      })}
    </TableRow>
  )
}

LibraTableRow.propTypes = {
  row: PropTypes.array.isRequired,
  columns: PropTypes.arrayOf(
    PropTypes.shape({
      keys: PropTypes.arrayOf(PropTypes.string).isRequired,
      label: PropTypes.string.isRequired,
      tooltip: PropTypes.string,
      renderCell: PropTypes.func.isRequired,
      indexes: PropTypes.arrayOf(PropTypes.number).isRequired,
    }),
  ).isRequired,
  isLoading: PropTypes.bool.isRequired,
  iconColumns: PropTypes.arrayOf(
    PropTypes.shape({
      renderCell: PropTypes.func.isRequired,
      keys: PropTypes.arrayOf(PropTypes.string).isRequired,
      indexes: PropTypes.arrayOf(PropTypes.number).isRequired,
    }),
  ).isRequired,
}

export default LibraTableRow
