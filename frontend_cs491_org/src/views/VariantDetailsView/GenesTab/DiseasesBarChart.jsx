import {
  Box,
  Alert,
  Card,
  CardContent,
  CardHeader,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Link,
  Stack,
  Tab,
  Tabs,
  Typography,
} from '@material-ui/core'
import { isEmpty } from 'ramda'
import React, { useEffect, useState } from 'react'
import ReactDOM from 'react-dom'
import { QueryClientProvider, useQueryClient } from 'react-query'
import ApexChart from 'src/components/ApexChart'

const extractDiseaseData = (diseaseData, activeTab) => {
  const data = [diseaseData.textmining, diseaseData.knowledge, diseaseData.experiments][activeTab].slice(0, 10)
  const xaxis = data.map((r) => r.doid_name)

  let yaxis = []
  let annotations = []

  switch (activeTab) {
    // Textmining
    case 0: {
      yaxis = data.map((r) => r.z_score / 2)
      annotations = data.map((r) => [{ label: 'ID', value: r.doid_id }])
      break
    }
    // Knowledge & Experiments
    case 1:
    case 2: {
      yaxis = data.map((r) => r.score)
      annotations = data.map((r) => [
        { label: 'Source', value: r.source },
        { label: 'Evidence', value: r.evidence },
      ])
      break
    }
  }

  yaxis = yaxis.map((r) => +r.toFixed(2))

  return { xaxis, yaxis, data, annotations }
}

const DiseaseBarChart = function ({ data, xaxis, yaxis, annotations }) {
  const queryClient = useQueryClient()

  const series = [{ name: 'Confidence', data: yaxis }]

  const handleClick = (_event, _chartContext, config) => {
    const { dataPointIndex } = config

    const url = data[dataPointIndex]?.url
    if (url) {
      window.open(url, '_blank').focus()
    }
  }

  const renderTooltip = (dataIndex) => {
    return (
      <Box display="flex" flexDirection="column">
        <Box>
          <span className="apexcharts-tooltip-text-y-label">Confidence:</span>
          <span className="apexcharts-tooltip-text-y-value">{yaxis[dataIndex]}</span>
        </Box>
        {annotations[dataIndex].map(({ label, value }, index) => (
          <Box key={index}>
            <span className="apexcharts-tooltip-text-y-label">{label}:</span>
            <span className="apexcharts-tooltip-text-y-value">{value}</span>
          </Box>
        ))}
      </Box>
    )
  }

  const options = {
    stroke: { show: false },
    plotOptions: {
      bar: { horizontal: true, barHeight: '30%', borderRadius: 4 },
    },
    xaxis: {
      type: 'category',
      categories: xaxis.map((value) => {
        if (value.length > 40) {
          return `${value.slice(0, 40)}…`
        }
        return value
      }),
      title: { text: 'Confidence' },
    },
    yaxis: {
      min: 0,
      max: 5,
      tickAmount: 1,
      labels: {
        maxWidth: 2000,
      },
    },
    grid: {
      padding: { left: 0, right: 0 },
    },
    tooltip: {
      x: { show: true },
      shared: true,
      intersect: false,
      custom: ({ series, seriesIndex, dataPointIndex, w }) => {
        if (series[seriesIndex]?.[dataPointIndex] === undefined) {
          return ''
        }

        const divEl = document.getElementById('apexchart-tooltip-container')

        ReactDOM.render(
          <QueryClientProvider client={queryClient}>
            <div style={{ fontFamily: w.config.chart.fontFamily }}>
              <div className="apexcharts-tooltip-title" style={{ fontSize: 13 }}>
                {xaxis[dataPointIndex]}
              </div>
              <div className="apexcharts-tooltip-series-group apexcharts-active" style={{ order: 1, display: 'flex' }}>
                <span className="apexcharts-tooltip-marker" style={{ backgroundColor: w.config.colors[0] }} />
                <div className="apexcharts-tooltip-text" style={{ fontSize: 12 }}>
                  <div className="apexcharts-tooltip-y-group">{renderTooltip(dataPointIndex)}</div>
                </div>
              </div>
            </div>
          </QueryClientProvider>,
          divEl,
        )

        return divEl.innerHTML
      },
    },
    chart: {
      events: {
        dataPointSelection: handleClick,
      },
    },
  }

  return <ApexChart type="bar" series={series} options={options} height="100%" />
}

const DiseaseBarChartContainer2 = function ({ disease, activeGene }) {
  const [activeTab, setActiveTab] = useState(0)

  useEffect(() => {
    setActiveTab(0)
  }, [disease])

  const tabsToShow = [
    !isEmpty(disease.textmining) && <Tab label="Text mining" key={0} value={0} />,
    !isEmpty(disease.knowledge) && <Tab label="Knowledge" key={1} value={1} />,
    !isEmpty(disease.experiments) && <Tab label="Experiments" key={2} value={2} />,
  ].filter(Boolean)

  const { data, xaxis, yaxis, annotations } = extractDiseaseData(disease, activeTab)

  // TODO: Refactor Card layout, it sucks as is
  return (
    <Card sx={{ height: '100%' }} component={Stack} direction="column">
      <CardHeader
        title={
          <Typography variant="h4" textAlign="center">
            {/* Make Link non-clickable if the url does not exist */}
            <Link target="_blank" href={disease.url} color="textPrimary" component={disease.url ? 'a' : 'span'}>
              DISEASES
            </Link>
          </Typography>
        }
      />
      <CardContent
        component={Stack}
        direction="column"
        justifyContent="center"
        alignItems={tabsToShow.length > 0 ? 'stretch' : 'center'}
        sx={{ flexGrow: 1 }}
      >
        {tabsToShow.length > 0 ? (
          <>
            <Tabs value={activeTab} onChange={(_e, value) => setActiveTab(value)} centered>
              {tabsToShow}
            </Tabs>
            <Box flexGrow={1} mx={3}>
              <DiseaseBarChart data={data} xaxis={xaxis} yaxis={yaxis} annotations={annotations} />
            </Box>
          </>
        ) : (
          <Alert severity="error" variant="outlined" textAlign="center">
            Could not find this gene in DISEASES database
          </Alert>
        )}
      </CardContent>
    </Card>
  )
}

const DiseasesTable = function ({ data, activeTab }) {
  const getConfidence = (row) => {
    return +(activeTab === 0 ? row.z_score / 2 : row.score).toFixed(2)
  }

  return (
    <TableContainer>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Phenotype</TableCell>
            {activeTab === 0 && <TableCell>DOID</TableCell>}
            {activeTab !== 0 && (
              <>
                <TableCell>Source</TableCell>
                <TableCell>Evidence</TableCell>
              </>
            )}
            <TableCell>Confidence</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {data.map((row, index) => (
            <TableRow key={index} sx={{ '&:last-child td, &:last-child th': { border: 0 } }}>
              <TableCell component="th" scope="row">
                {row.url ? (
                  <Link href={row.url} target="_blank" color="textPrimary">
                    {row.doid_name}
                  </Link>
                ) : (
                  row.doid_name
                )}
              </TableCell>
              {activeTab === 0 && <TableCell>{row.doid_id}</TableCell>}
              {activeTab !== 0 && (
                <>
                  <TableCell>{row.source}</TableCell>
                  <TableCell>{row.evidence}</TableCell>
                </>
              )}
              <TableCell>{getConfidence(row)}</TableCell>
            </TableRow>
          ))}
          {/* {omim.phenotypes.map((row) => (
            <TableRow key={row.name} sx={{ '&:last-child td, &:last-child th': { border: 0 } }}>
              <TableCell component="th" scope="row">
                {row.description}
              </TableCell>
              <TableCell>{row.omim_number}</TableCell>
              <TableCell>{INHERITANCE_MODE_TO_HUMAN_READABLE[row.inheritance_mode]}</TableCell>
              <TableCell>{row.mapping_key}</TableCell>
            </TableRow>
          ))} */}
        </TableBody>
      </Table>
    </TableContainer>
  )
}

const DiseaseBarChartContainer = function ({ disease, activeGene }) {
  const [activeTab, setActiveTab] = useState(0)

  useEffect(() => {
    setActiveTab(0)
  }, [disease])

  const tabsToShow = [
    !isEmpty(disease.textmining) && <Tab label="Text mining" key={0} value={0} />,
    !isEmpty(disease.knowledge) && <Tab label="Knowledge" key={1} value={1} />,
    !isEmpty(disease.experiments) && <Tab label="Experiments" key={2} value={2} />,
  ].filter(Boolean)

  const { data, xaxis, yaxis, annotations } = extractDiseaseData(disease, activeTab)

  // TODO: Refactor Card layout, it sucks as is
  return (
    <Card sx={{ height: '100%' }} component={Stack} direction="column">
      <CardHeader
        title={
          <Typography variant="h4" textAlign="center">
            {/* Make Link non-clickable if the url does not exist */}
            <Link target="_blank" href={disease.url} color="textPrimary" component={disease.url ? 'a' : 'span'}>
              DISEASES
            </Link>
          </Typography>
        }
      />
      <CardContent
        component={Stack}
        direction="column"
        justifyContent="center"
        alignItems={tabsToShow.length > 0 ? 'stretch' : 'center'}
        sx={{ flexGrow: 1 }}
      >
        {tabsToShow.length > 0 ? (
          <>
            <Tabs value={activeTab} onChange={(_e, value) => setActiveTab(value)} centered>
              {tabsToShow}
            </Tabs>
            <Box p={1} />
            <Box flexGrow={1} mx={3}>
              <DiseasesTable data={data} xaxis={xaxis} yaxis={yaxis} annotations={annotations} activeTab={activeTab} />
            </Box>
          </>
        ) : (
          <Alert severity="error" variant="outlined" textAlign="center">
            Could not find this gene in DISEASES database
          </Alert>
        )}
      </CardContent>
    </Card>
  )
}

export default DiseaseBarChartContainer
