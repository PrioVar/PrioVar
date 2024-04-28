import { TableCell, TableHead, TableRow, TableSortLabel } from '@material-ui/core'
import { makeStyles } from '@material-ui/styles'
import PropTypes from 'prop-types'
import { times } from 'ramda'
import React from 'react'

// TODO: Find a better solution to the box shadowing problem

const useStyles = makeStyles((theme) => ({
  row: {
    '& > th': {
      boxShadow: `${[
        '0px 3px 5px -1px rgb(145 158 171 / 20%)',
        '0px 5px 8px 0px rgb(145 158 171 / 14%)',
        '0px 1px 14px 0px rgb(145 158 171 / 12%)',
      ].join(',')} !important`,
      borderRadius: '0 !important',
      padding: `${theme.spacing(0, 0, 1.5, 0.5)} !important`,
      margin: '0 !important',
      clipPath: 'polygon(0% 0%, 100% 0%, 100% 200%, 0% 200%)',
    },
  },
  tableHeadShadow: {
    boxShadow: `${[
      '0px 3px 5px -5px rgb(145 158 171 / 20%)',
      '0px 5px 8px -8px rgb(145 158 171 / 14%)',
      '0px 1px 14px -14px rgb(145 158 171 / 12%)',
    ].join(',')} !important`,
  },
}))

function LibraTableHead({ sort, onChangeSort, columns, iconColumnCount }) {
  const classes = useStyles()

  const createHandleChangeSort = (keys) => (_event) => {
    const isCurrentlyAsc = sort.columnKey === keys[0] && sort.direction === 'asc'
    onChangeSort({
      columnKey: keys[0],
      direction: isCurrentlyAsc ? 'desc' : 'asc',
    })
  }

  return (
    <TableHead>
      <TableRow className={classes.row}>
        {/* empty cell for icon columns */}
        {times(
          (i) => (
            <TableCell key={`empty-cell.${i}`} />
          ),
          iconColumnCount,
        )}
        {columns.map((column, columnIndex) => {
          const isSorted = column.keys.includes(sort.columnKey)

          return (
            <TableCell key={columnIndex} sortDirection={isSorted ? sort.direction : false} sx={{ p: 0, m: 0 }}>
              <TableSortLabel
                active={isSorted}
                direction={isSorted ? sort.direction : 'asc'}
                onClick={createHandleChangeSort(column.keys)}
              >
                {column.label}
              </TableSortLabel>
            </TableCell>
          )
        })}
      </TableRow>
    </TableHead>
  )
}

LibraTableHead.propTypes = {
  onChangeSort: PropTypes.func.isRequired,
  sort: PropTypes.shape({
    columnKey: PropTypes.string.isRequired,
    direction: PropTypes.oneOf(['asc', 'desc']),
  }),
  columns: PropTypes.arrayOf(
    PropTypes.shape({
      keys: PropTypes.arrayOf(PropTypes.string).isRequired,
      label: PropTypes.string.isRequired,
      tooltip: PropTypes.string,
      renderCell: PropTypes.func.isRequired,
      indexes: PropTypes.arrayOf(PropTypes.number).isRequired,
    }),
  ).isRequired,
  iconColumnCount: PropTypes.number.isRequired,
}

export default LibraTableHead
