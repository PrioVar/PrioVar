import { Chip, Link, Stack, TableCell, TableRow, Tooltip, Typography } from '@material-ui/core'
import { alpha } from '@material-ui/core/styles'
import FlareRoundedIcon from '@material-ui/icons/FlareRounded'
import { makeStyles } from '@material-ui/styles'
import { orderBy } from 'lodash'
import MUIDataTable from 'mui-datatables'
import React from 'react'

const useStyles = makeStyles((theme) => ({
  tableRow: (props) => {
    const bg2 = theme.palette[props.color].main || theme.palette[props.color][400]
    const bg = alpha(bg2, props.isCanonical ? 0.3 : 0.16)

    return {
      color: theme.palette[props.color][theme.palette.mode === 'light' ? 'dark' : 'light'],
      backgroundColor: bg,
      borderBottom: '1px solid #dde',
    }
  },
}))

const impactOrder = ['MODIFIER', 'LOW', 'MODERATE', 'HIGH']

const sortTranscripts = (transcripts) => {
  return orderBy(transcripts, (t) => [impactOrder.indexOf(t.Impact), t.Is_Canonical, t.Gene_Symbol, t.Transcript_Id], [
    'desc',
    'desc',
    'asc',
    'asc',
  ])
}

const defaultRender = (value) => {
  return (
    <Typography noWrap variant="body2">
      {value}
    </Typography>
  )
}

const TranscriptFactorsTableRow = function ({ row, columns }) {
  const color = {
    LOW: 'primary',
    MODERATE: 'warning',
    HIGH: 'error',
    MODIFIER: 'grey',
  }[row.Impact]

  const classes = useStyles({ color, isCanonical: row.Is_Canonical })
  /*
  let background = {
    LOW: '#e9fcdf',
    MODERATE: '#fff7da',
    HIGH: '#ffe8e2',
    MODIFIER: '#fafafa',
  }[row.Impact]

  if (row.Is_Canonical) {
    background = chroma(background).saturate().css()
  }
    const scaleValue = {
    LOW: 1,
    MODERATE: 0.35,
    HIGH: 0,
    MODIFIER: 0.5,
  }[row.Impact]

  const background = COLOR_SCALE(scaleValue).css()
   */

  return (
    <TableRow className={classes.tableRow}>
      {columns.map((column) => {
        const { render = defaultRender } = column.options
        const value = row[column.name]

        return <TableCell key={column.name}>{render(value, row)}</TableCell>
      })}
    </TableRow>
  )
}

const TranscriptFactorsTab = function ({ data }) {
  const sortedData = sortTranscripts(data)

  const createColumn = (name, label, options = {}) => ({
    name,
    label,
    options: { filter: false, sort: false, ...options },
  })

  const columns = [
    createColumn('Is_Canonical', ' ', {
      render(isCanonical) {
        if (!isCanonical) {
          return null
        }

        return (
          <Tooltip title="Canonical">
            <FlareRoundedIcon />
          </Tooltip>
        )
      },
    }),
    createColumn('Allele', 'Allele'),
    createColumn('Consequences', 'Consequence', {
      render: (consequences) => (
        <Stack direction="row" spacing={0.5}>
          {consequences.map((consequence) => (
            <Chip key={consequence} variant="filled" color="default" label={consequence} />
          ))}
        </Stack>
      ),
    }),
    // TODO: Calculate impact in VEP
    createColumn('Impact', 'Impact'),
    createColumn('HGVSc', 'HGVSc', {
      render: (hgvsC) => {
        if (!hgvsC) return ''
        const [_ensemblId, rest] = hgvsC.split(':')
        return defaultRender(rest)
      },
    }),
    createColumn('HGVSp', 'HGVSp', {
      render: (hgvsP) => {
        if (!hgvsP) return ''
        const [_ensemblId, rest] = hgvsP.split(':')
        return defaultRender(decodeURIComponent(rest))
      },
    }),
    createColumn('Gene_Symbol', 'Gene Symbol', {
      render: (value) => (
        <Link color="textPrimary" target="_blank" href={`https://pubchem.ncbi.nlm.nih.gov/gene/${value}/human`}>
          {value}
        </Link>
      ),
    }),
    createColumn('Gene_Id', 'Gene ID', {
      render: (value) => (
        <Link color="textPrimary" target="_blank" href={`https://www.ensembl.org/Homo_sapiens/Gene/Summary?g=${value}`}>
          {value}
        </Link>
      ),
    }),
    createColumn('Transcript_Id', 'Transcript ID', {
      render: (value) => (
        <Link
          color="textPrimary"
          target="_blank"
          href={`https://ensembl.org/Homo_sapiens/Transcript/Summary?t=${value}`}
        >
          {value}
        </Link>
      ),
    }),
  ]

  return (
    <MUIDataTable
      title=""
      data={sortedData}
      columns={columns}
      components={{
        TableToolbar: () => null,
      }}
      options={{
        selectableRows: 'none',
        pagination: false,
        customRowRender: (_rowArray, dataIndex) => {
          const rowObject = sortedData[dataIndex]
          return <TranscriptFactorsTableRow key={dataIndex} row={rowObject} columns={columns} />
        },
      }}
    />
  )
}

export default TranscriptFactorsTab
