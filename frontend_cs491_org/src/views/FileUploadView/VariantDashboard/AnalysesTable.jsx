import { Fab, Box, Card, CardHeader, Grid, CardActions, TableRow, Button, Chip } from '@material-ui/core'
import MUIDataTable from 'mui-datatables'
import React from 'react'
import { fDateTime } from 'src/utils/formatTime'
import { useNavigate, useParams } from 'react-router-dom'
import { makeStyles } from '@material-ui/styles'
import AddIcon from '@material-ui/icons/Add'
import ArrowForwardIcon from '@material-ui/icons/ArrowForward'
import JobStateStatus from '../../common/JobStateStatus'
import { API_BASE_URL } from '../../../constants'

const useStyles = makeStyles((theme) => ({
  analysesTable: {
    height: '100%',
    bgcolor: 'text.secondary',
  },
}))

const GoToSnpAnalysis = function ({ fileId, sampleName }) {
  const navigate = useNavigate()

  const handleClick = () => {
    navigate(`/priovar/variants/${fileId}/${sampleName}`)
  }

  return (
    <Button variant="contained" onClick={handleClick} size="small">
      <ArrowForwardIcon />
    </Button>
  )
}

const GoToCnvAnalysis = function ({ sampleName, analysisId }) {
  return (
    <Button
      variant="contained"
      size="small"
      download={`${sampleName}.cnv.txt`}
      href={`${API_BASE_URL}/cnv/${analysisId}`}
      target="_blank"
    >
      <ArrowForwardIcon />
    </Button>
  )
}

const AnalysesTable = function ({ data, sampleName, fileId, onCreate }) {
  const classes = useStyles()

  const COLUMNS = [
    {
      name: 'created_at',
      label: 'Started At',
      options: {
        filter: false,
        sort: true,
        customBodyRenderLite(dataIndex) {
          const row = data[dataIndex]
          if (!row) {
            return null
          }

          return fDateTime(row.created_at)
        },
      },
    },
    {
      name: 'updated_at',
      label: 'Updated At',
      options: {
        filter: false,
        sort: true,
        customBodyRenderLite(dataIndex) {
          const row = data[dataIndex]
          if (!row) {
            return null
          }

          return fDateTime(row.updated_at)
        },
      },
    },
    {
      name: 'type',
      label: 'Type',
      options: {
        filter: false,
        sort: true,
        customBodyRenderLite: (dataIndex) => {
          const row = data[dataIndex]
          switch (row?.type) {
            case 'vcf':
              return <Chip label={'Variant Annotation Analysis'} />
            case 'bwa':
              return <Chip label={'Alignment: BWA MEM v1'} />
            case 'gatk':
              return <Chip label={'SNP: GATK v4'} />
            case 'xhmm':
              return <Chip label={'CNV: XHMM'} />
            case 'xhmm+decont':
              return <Chip label={'CNV: XHMM + DECoNT'} />
            default:
              return null
          }
        },
      },
    },
    {
      name: 'status',
      label: 'Status',
      options: {
        filter: false,
        sort: true,
        customBodyRenderLite: (dataIndex) => {
          const row = data[dataIndex]
          if (!row) {
            return null
          }

          return <JobStateStatus status={row.status} isAnalysis />
        },
      },
    },
    {
      name: 'go',
      label: ' ',
      options: {
        filter: false,
        sort: false,
        customBodyRenderLite: (dataIndex) => {
          const row = data[dataIndex]
          if (row?.status !== 'DONE') {
            return null
          }
          switch (row?.type) {
            case 'vcf':
              return <GoToSnpAnalysis fileId={fileId} sampleName={sampleName} />
            case 'gatk':
              return <GoToSnpAnalysis fileId={row.vcf_id} sampleName={sampleName} />
            case 'xhmm':
            case 'xhmm+decont':
              return <GoToCnvAnalysis analysisId={row.id} sampleName={sampleName} />
            default:
              return null
          }
        },
      },
    },
  ]

  return (
    <Card sx={{ height: '100%' }}>
      <CardHeader
        title="Analyses"
        action={
          <Fab color="primary" size="small" onClick={onCreate}>
            <AddIcon />
          </Fab>
        }
      />
      <Box p={1} />
      <MUIDataTable
        title=""
        data={data}
        columns={COLUMNS}
        options={{
          selectableRows: 'none',
          sortOrder: { name: 'created_at', direction: 'desc' },
          textLabels: {
            body: {
              noMatch: `No analyses found.`,
            },
          },
        }}
        className={classes.analysesTable}
        components={{
          TableToolbar: () => null,
          TableFooter: () => null,
        }}
      />
    </Card>
  )
}

export default AnalysesTable
