import {
  Box,
  CircularProgress,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TablePagination,
  TableRow,
  Typography,
} from '@material-ui/core'
import { makeStyles, withStyles } from '@material-ui/styles'
import PropTypes from 'prop-types'
import React from 'react'
import usePrevious from '../../hooks/usePrevious'
import LibraTableHead from './TableHead'
import LibraTableRow from './TableRow'

const StyledPagination = withStyles({
  toolbar: {
    height: '40px',
    minHeight: '40px',
  },
})(TablePagination)

const useStyles = makeStyles({
  root: {
    width: '100%',
    display: 'flex',
    flexDirection: 'column',
  },
  table: {
    '& td': {
      /* Each cell should grow equally */
      width: '1%',
      '&.collapsed': {
        /* But "collapsed" cells should be as small as possible */
        width: '0.0000000001%',
      },
    },
  },
  tableContainer: {
    maxHeight: (props) => (props.isPaginated ? 'calc(100vh - 160px)' : '100%'),
  },
})

const applyColumnConfig = (columnConfig, columns) => {
  const calculateColumnIndex = (keys) => {
    return keys.map((key) => columns.indexOf(key))
  }

  return columnConfig
    .filter((config) => !config.hidden)
    .map((config) => ({
      ...config,
      indexes: calculateColumnIndex(config.keys),
    }))
}

function LibraTable({
  rows,
  columns,
  columnConfig,
  page,
  pageSize,
  pageSizeOptions,
  onChangePage,
  onChangePageSize,
  totalRowCount,
  onChangeSort,
  sort,
  isLoading,
  toolbarPanelComponent,
  isPaginated = false,
  iconColumns,
}) {
  const classes = useStyles({ isPaginated })
  const previousRows = usePrevious(rows)
  const previousColumns = usePrevious(columns)
  const previousTotalRowCount = usePrevious(totalRowCount)

  const columnsToDisplay = isLoading
    ? applyColumnConfig(columnConfig, previousColumns)
    : applyColumnConfig(columnConfig, columns)
  const rowsToDisplay = isLoading ? previousRows : rows
  const totalRowCountToDisplay = isLoading ? previousTotalRowCount : totalRowCount

  const emptyRows = 0
  // TODO: Fix broken calculation
  // const emptyRows = pageSize - Math.min(pageSize, rows.length - page * pageSize)

  const isEmpty = !rowsToDisplay || rowsToDisplay.length === 0

  return (
    <Paper className={classes.root} elevation={5}>
      <TableContainer className={classes.tableContainer}>
        <Table className={classes.table} stickyHeader size="medium">
          <LibraTableHead
            classes={classes}
            sort={sort}
            onChangeSort={onChangeSort}
            columns={columnsToDisplay}
            iconColumnCount={iconColumns.length}
          />
          <TableBody>
            {isEmpty && (
              <TableRow>
                <TableCell colSpan={isLoading ? 100 : columns.length}>
                  <Box
                    display="inline-flex"
                    justifyContent="center"
                    alignItems="center"
                    sx={{ width: '100%', height: '15vh' }}
                  >
                    {isLoading ? (
                      <CircularProgress size="10vh" />
                    ) : (
                      <Typography variant="body2">Sorry, we could not find any records...</Typography>
                    )}
                  </Box>
                </TableCell>
              </TableRow>
            )}
            {rowsToDisplay.map((row) => (
              <LibraTableRow
                row={row}
                columns={columnsToDisplay}
                isLoading={isLoading}
                iconColumns={applyColumnConfig(iconColumns, columns)}
              />
            ))}
            {emptyRows > 0 && (
              <TableRow style={{ height: 53 * emptyRows }}>
                <TableCell colSpan={6} />
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
      {isPaginated && (
        <StyledPagination
          id="table-pagination"
          rowsPerPageOptions={pageSizeOptions}
          count={totalRowCountToDisplay}
          rowsPerPage={pageSize}
          page={page}
          onPageChange={(_e, page) => onChangePage(page)}
          onRowsPerPageChange={(e) => onChangePageSize(e.target.value)}
          sx={{ alignSelf: 'flex-start' }}
        />
      )}
    </Paper>
  )
}

LibraTable.propTypes = {
  rows: PropTypes.array.isRequired,
  columns: PropTypes.arrayOf(PropTypes.string).isRequired,
  columnConfig: PropTypes.arrayOf(
    PropTypes.shape({
      keys: PropTypes.arrayOf(PropTypes.string).isRequired,
      label: PropTypes.string.isRequired,
      tooltip: PropTypes.string,
      renderCell: PropTypes.func.isRequired,
      collapse: PropTypes.bool,
      hidden: PropTypes.bool,
    }),
  ).isRequired,
  totalRowCount: PropTypes.number.isRequired,
  isLoading: PropTypes.bool.isRequired,
  page: PropTypes.number.isRequired,
  pageSize: PropTypes.number.isRequired,
  pageSizeOptions: PropTypes.arrayOf(PropTypes.number).isRequired,
  sort: PropTypes.shape({
    columnIndex: PropTypes.number.isRequired,
    direction: PropTypes.oneOf(['asc', 'desc']),
  }),
  onChangePage: PropTypes.func.isRequired,
  onChangePageSize: PropTypes.func.isRequired,
  onChangeSort: PropTypes.func.isRequired,
  toolbarPanelComponent: PropTypes.node.isRequired,
  isPaginated: PropTypes.bool.isRequired,
  iconColumns: PropTypes.arrayOf(
    PropTypes.shape({
      renderCell: PropTypes.func.isRequired,
      keys: PropTypes.string.isRequired,
    }),
  ).isRequired,
}

export default LibraTable
