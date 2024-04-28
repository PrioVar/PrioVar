import closeFill from '@iconify-icons/eva/close-fill'
import { Icon } from '@iconify/react'
import {
  Box,
  Button,
  Checkbox,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Divider,
  FormControl,
  FormControlLabel,
  FormLabel,
  IconButton,
  Radio,
  RadioGroup,
  Stack,
  TextField,
  Typography,
  Tabs,
  Tab,
} from '@material-ui/core'
import AccountTreeRoundedIcon from '@material-ui/icons/AccountTreeRounded'
import FilterListIcon from '@material-ui/icons/FilterList'
import AddIcon from '@material-ui/icons/Add'
import ArrowBackIosIcon from '@material-ui/icons/ArrowBackIos'
import BallotIcon from '@material-ui/icons/Ballot'
import PictureAsPdfIcon from '@material-ui/icons/PictureAsPdf'
import { createStyles, makeStyles, withStyles } from '@material-ui/styles'

import React, { useRef, useState, useEffect, useMemo } from 'react'
import ReactDOM from 'react-dom'
import { useNavigate, useParams } from 'react-router-dom'
import { useReactToPrint } from 'react-to-print'
import { useDispatch, Provider } from 'react-redux'
import { useFormik } from 'formik'
import PropTypes from 'prop-types'
import { createStore } from 'redux'
import { v4 as uuidv4 } from 'uuid'
import * as yup from 'yup'

import { HPO_OPTIONS, FILTER_OPTIONS, NAVBAR_CONFIG } from 'src/constants'
import Page from 'src/components/Page'
import Tags from 'src/components/Tags'
import { useTrio } from '../../api/vcf/trio'
import { useHpo, useSampleMetadata } from '../../api/vcf'
import { useVariants } from 'src/api/variant'

import HpoPanel from './HpoPanel'
import Sidebar from './Sidebar'
import NavbarRoutes from './NavbarRoutes'
import VariantsTable from './VariantsTable'
import VariantsTableToolbar from './Toolbar'

import { actions as variantFiltersActions, reducer as variantFiltersReducer } from 'src/redux/slices/variantFilters'

const getNewTableTitle = ({ variant, sampleName, filterName = '' }) => {
  switch (variant) {
    case 'kh1':
      return `High Risk Variants of ${sampleName} detected by PrioVarAI™`
    case 'kh2':
      return `Medium Risk Variants of ${sampleName} detected by PrioVarAI™`
    case 'kh3':
      return `Target Area Variants of ${sampleName} detected by PrioVarAI™`
    case 'kh4':
      return `Variants supporting clinical evidence for ${sampleName} detected by PrioVarAI™`
    case 'Low Coverage':
    case 'ACMG Incidentals':
      return `${filterName === '' ? 'Predefined' : filterName} genes for sample ${sampleName}`
    case 'kh6':
      return `Non-benign variants for sample ${sampleName}`
    case 'standard':
    default:
      return `All detected variants of ${sampleName}`
  }
}

const mapSelectedVariantsToMultiPosStr = (selectedVariants = []) => {
  return selectedVariants.map(({ chrom, pos }) => `${chrom}:${pos}-${pos}`).join(',')
}

const CreateTableDialog = function ({ open, onClose, onClickCreateTable }) {
  const { fileId, sampleName } = useParams()
  const [hpoList, setHpoList] = useState([])
  const [predefinedFilter, setPredefinedFilter] = useState('')
  const { data: hpoData = [] } = useHpo({
    fileId,
    sampleName,
  })

  /*
  useEffect(() => {
    const newHpoList = hpoData.map((hpoId) => HPO_MAP[hpoId])
    setHpoList(newHpoList)
  }, [hpoData])
  */
  const formik = useFormik({
    initialValues: {
      title: getNewTableTitle({ variant: 'standard' }),
      variant: 'standard',
      hpo: {
        mode: 'any',
        similar: false,
      },
    },
    validationSchema: yup.object({
      title: yup.string().required(),
      variant: yup
        .mixed()
        .oneOf(['standard', 'kh1', 'kh2', 'kh3', 'kh4', 'Low Coverage', 'ACMG Incidentals', 'kh6'])
        .required(),
      hpo: yup
        .object({
          similar: yup.bool().required(),
          mode: yup.mixed().oneOf(['all', 'any', 'most']).required(),
        })
        .required(),
    }),
    onSubmit: (values) => {
      const hpoIds = hpoList.map((hpo) => hpo.value)
      if (FILTER_OPTIONS.includes(values.variant)) {
        values.variant = 'kh5'
      }

      onClose()
      onClickCreateTable({
        ...values,
        hpo: {
          ...values.hpo,
          ids: hpoIds,
        },
        predefinedFilter: predefinedFilter,
      })
    },
  })

  const handleChangeVariant = (e, value) => {
    formik.values.title = getNewTableTitle({ variant: value, sampleName, filterName: value })
    formik.handleChange('variant')(e, value)
    if (FILTER_OPTIONS.includes(value)) {
      setPredefinedFilter(value)
    }
  }

  const handleHpoChange = (newData) => {
    setHpoList(newData)
  }

  return (
    <Dialog open={open} onClose={onClose} maxWidth={false} PaperProps={{ sx: { width: 720 } }}>
      <form onSubmit={formik.handleSubmit}>
        <DialogTitle>Create a new table</DialogTitle>
        <DialogContent>
          <Box p={1} />
          <DialogContentText>Create a new table based on the following options.</DialogContentText>
          <Box p={1} />
          <TextField
            margin="dense"
            fullWidth
            name="title"
            label="Title"
            value={formik.values.title}
            onChange={formik.handleChange}
            error={formik.touched.title && Boolean(formik.errors.title)}
            helperText={formik.touched.title && formik.errors.title}
          />
          <Box p={1} />
          <FormControl component="fieldset" sx={{ display: 'flex' }}>
            <FormLabel component="legend">Starting filters</FormLabel>
            <Box p={1} />
            <RadioGroup row name="variant" value={formik.values.variant} onChange={handleChangeVariant}>
              <FormControlLabel
                label="None"
                value="standard"
                control={<Radio color="primary" />}
                labelPlacement="top"
              />
              {/*               <FormControlLabel
                label="High Risk Variants"
                value="kh1"
                control={<Radio color="error" />}
                labelPlacement="top"
              /> */}
              <FormControlLabel
                label="Non-benign Variants"
                value="kh6"
                control={<Radio color="info" />}
                labelPlacement="top"
              />
              {/*  <FormControlLabel
                label="Medium Risk Variants"
                value="kh2"
                control={<Radio color="warning" />}
                labelPlacement="top"
              /> */}
              {/*    <FormControlLabel
                label="Target Area Variants"
                value="kh3"
                control={<Radio color="info" />}
                labelPlacement="top"
              /> */}
              {/* <FormControlLabel
                label="Variants Supporting Clinical Evidence"
                value="kh4"
                control={<Radio color="info" />}
                labelPlacement="top"
              /> */}
              {FILTER_OPTIONS.map((filter) => (
                <FormControlLabel label={filter} value={filter} control={<Radio color="info" />} labelPlacement="top" />
              ))}
            </RadioGroup>
          </FormControl>
          {formik.values.variant === 'kh4' && (
            <FormControl component="fieldset" sx={{ display: 'flex' }} margin="normal">
              <Tags
                title={`Selected diseases for ${sampleName}`}
                options={HPO_OPTIONS}
                value={hpoList}
                onChange={handleHpoChange}
                sx={{ maxWidth: '100%' }}
              />
              <Box p={1} />
              <FormControlLabel
                control={<Checkbox />}
                label="Include similar diseases"
                name="hpo.similar"
                value={formik.values.hpo.similar}
                onChange={formik.handleChange}
              />
              <Box p={1} />
              <FormLabel>Include any, all or majority of the diseases above</FormLabel>
              <Box p={1} />
              <RadioGroup row name="hpo.mode" value={formik.values.hpo.mode} onChange={formik.handleChange}>
                <FormControlLabel label="Any" value="any" control={<Radio color="info" />} labelPlacement="top" />
                <FormControlLabel label="All" value="all" control={<Radio color="info" />} labelPlacement="top" />
                <FormControlLabel label="Majority" value="most" control={<Radio color="info" />} labelPlacement="top" />
              </RadioGroup>
            </FormControl>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={onClose} color="secondary">
            Cancel
          </Button>
          <Button onClick={formik.handleSubmit} color="primary">
            Create
          </Button>
        </DialogActions>
      </form>
    </Dialog>
  )
}

CreateTableDialog.propTypes = {
  open: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  onClickCreateTable: PropTypes.func.isRequired,
}

const TrioView = function () {
  const { fileId } = useParams()
  const { status, data } = useTrio({ fileId })
  const navigate = useNavigate()

  if (status !== 'success') {
    return (
      <Stack justifyContent="center">
        <CircularProgress />
      </Stack>
    )
  }

  const { mother_file, mother_sample_name, father_file, father_sample_name } = data

  const createHandleClick = (fileId, sampleName) => () => {
    navigate(`/priovar/variants/${fileId}/${sampleName}`)
  }

  return (
    <Stack direction="row" spacing={2}>
      <Stack direction="column">
        <Typography gutterBottom variant="subtitle2" sx={{ color: 'text.secondary' }}>
          Mother
        </Typography>
        <Button
          variant="outlined"
          color="error"
          sx={{ height: '100%' }}
          onClick={mother_file && createHandleClick(mother_file, mother_sample_name)}
        >
          {mother_sample_name || '?'}
        </Button>
      </Stack>
      <Stack direction="column">
        <Typography gutterBottom variant="subtitle2" sx={{ color: 'text.secondary' }}>
          Father
        </Typography>
        <Button
          variant="outlined"
          color="info"
          sx={{ height: '100%' }}
          onClick={father_file && createHandleClick(father_file, father_sample_name)}
        >
          {father_sample_name || '?'}
        </Button>
      </Stack>
    </Stack>
  )
}

const useStyles = makeStyles(
  createStyles({
    '@global': {
      '@media print': {
        body: {
          height: 'initial !important',
          overflow: 'initial !important',
        },
      },
      '@page': {
        size: 'auto',
        margin: '20mm',
      },
    },
  }),
)

function TabPanel(props) {
  const { children, value, index, ...other } = props

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 1 }}>
          <Typography>{children}</Typography>
        </Box>
      )}
    </div>
  )
}

TabPanel.propTypes = {
  children: PropTypes.node,
  index: PropTypes.number.isRequired,
  value: PropTypes.number.isRequired,
}

function a11yProps(index) {
  return {
    id: `simple-tab-${index}`,
    'aria-controls': `simple-tabpanel-${index}`,
  }
}

const VarTab = withStyles((theme) => ({
  root: {
    padding: '5px',
    '&:hover': {
      backgroundColor: 'rgb(255 255 255 / 10%)',
      opacity: 1,
    },
    '&$selected': {
      backgroundColor: 'transparent',
    },
  },
  selected: {},
}))((props) => <Tab disableRipple {...props} />)

const VariantsView = function () {
  useStyles()
  const dispatch = useDispatch()
  const { fileId, sampleName } = useParams()
  const sampleMetadata = useSampleMetadata({ fileId, sampleName })
  const [isHpoPanelOpen, setIsHpoPanelOpen] = useState(false)
  const [isToolbarPanelOpen, setIsToolbarPanelOpen] = useState(false)
  const [isSelectedVariantsMode, setIsSelectedVariantsMode] = useState(false)
  const [isCreateNewDialogOpen, setIsCreateNewDialogOpen] = useState(false)
  const [value, setValue] = React.useState(0)
  const tablesRef = useRef()
  const handlePrint = useReactToPrint({
    content: () => tablesRef.current,
  })

  const handleTabChange = (event, newValue) => {
    setValue(newValue)
  }

  // TODO: Refactor
  const handleClickCreateTable = ({ variant, title, hpo, predefinedFilter }) => {
    const newTable = { id: uuidv4(), variant, title, selectedVariants: [], hpo, predefinedFilter }
    sampleMetadata.addNewTable(newTable)
  }

  const handleClickDeleteTable = (tableId) => {
    sampleMetadata.removeTable(tableId)
  }

  const handleCreatePdf = () => {
    handlePrint()
  }

  function CloseButton(props) {
    const { tableId, index } = props

    return (
      <div>
        {index !== 0 && !isSelectedVariantsMode && (
          <IconButton onClick={() => handleClickDeleteTable(tableId)}>
            <Icon icon={closeFill} width={20} height={20} />
          </IconButton>
        )}
      </div>
    )
  }

  const tables = useMemo(() => sampleMetadata.query.data?.tables ?? [], [sampleMetadata.query.data])

  const [sortOrder, setSortOrder] = useState({
    columnKey: 'ACMG',
    direction: 'desc',
  })
  const [page, setPage] = useState(0)
  const [pageSize, setPageSize] = useState(50)

  const { status, data = {} } = useVariants({
    fileId,
    sampleName,
    page,
    pageSize,
    sortBy: sortOrder.columnKey,
    sortDirection: sortOrder.direction,
  })

  useEffect(() => {
    if (tables[value]) {
      if (tables[value].variant === 'standard') {
        dispatch(variantFiltersActions.setKh1(false))
        dispatch(variantFiltersActions.setKh2(false))
        dispatch(variantFiltersActions.setKh3(false))
        dispatch(variantFiltersActions.setHpo(null))
        dispatch(variantFiltersActions.setPredefined(null))
        dispatch(variantFiltersActions.setNonBenign(false))
      }
      if (tables[value].variant === 'kh1') {
        dispatch(variantFiltersActions.setKh1(true))
      }
      if (tables[value].variant === 'kh2') {
        dispatch(variantFiltersActions.setKh2(true))
      }
      if (tables[value].variant === 'kh3') {
        dispatch(variantFiltersActions.setKh3(true))
      }
      if (tables[value].variant === 'kh4') {
        dispatch(
          variantFiltersActions.setHpo({
            ids: tables[value].hpo.ids,
            similar: tables[value].hpo.similar,
            mode: tables[value].hpo.mode,
          }),
        )
      }
      if (tables[value].variant === 'kh5') {
        dispatch(
          variantFiltersActions.setPredefined({
            predefinedFilter: tables[value].predefinedFilter,
          }),
        )
      }
      if (tables[value].variant === 'kh6') {
        dispatch(variantFiltersActions.setNonBenign(true))
        dispatch(variantFiltersActions.setKh3(true))
      }
    }
    // TODO: Adding hpoList makes infinite loop
  }, [dispatch, value, tables])

  useEffect(() => {
    if (isSelectedVariantsMode && tables[value]) {
      // FIXME: Find an elegant solution to this problem
      const multiPosStr = mapSelectedVariantsToMultiPosStr(tables[value].selectedVariants) || 'chr0:0-0'
      dispatch(variantFiltersActions.setMultipos(multiPosStr))
    } else {
      dispatch(variantFiltersActions.setMultipos(''))
    }

    setPage(0)
  }, [dispatch, isSelectedVariantsMode, tables, value])

  return (
    <Page title="Variants | PrioVar">
      <CreateTableDialog
        onClose={() => setIsCreateNewDialogOpen(false)}
        open={isCreateNewDialogOpen}
        onClickCreateTable={handleClickCreateTable}
      />
      <Sidebar isOpen={isHpoPanelOpen} onClose={() => setIsHpoPanelOpen(false)}>
        {isHpoPanelOpen && <HpoPanel />}
      </Sidebar>
      <Sidebar isOpen={isToolbarPanelOpen} onClose={() => setIsToolbarPanelOpen(false)} anchor="left">
        {isToolbarPanelOpen && (
          <VariantsTableToolbar variant={tables[value].variant} onToolbarClose={() => setIsToolbarPanelOpen(false)} />
        )}
      </Sidebar>
      {ReactDOM.createPortal(
        <Stack direction="row" spacing={2}>
          {sampleName && (
            <>
              <NavbarRoutes navConfig={NAVBAR_CONFIG} params={{ fileId, sampleName }} />
              <Button
                variant="text"
                color="error"
                startIcon={isSelectedVariantsMode ? <ArrowBackIosIcon /> : <BallotIcon />}
                onClick={() => setIsSelectedVariantsMode(!isSelectedVariantsMode)}
              >
                {isSelectedVariantsMode ? 'Back' : 'Selected Variants'}
              </Button>
              <Divider orientation="vertical" flexItem />
              <Button
                variant="text"
                color="secondary"
                startIcon={<AccountTreeRoundedIcon />}
                onClick={() => setIsHpoPanelOpen(true)}
              >
                HPO
              </Button>
              <Divider orientation="vertical" flexItem />
              <Button
                variant="text"
                color="info"
                startIcon={<FilterListIcon />}
                onClick={() => setIsToolbarPanelOpen(true)}
              >
                Filters
              </Button>
              <Divider orientation="vertical" flexItem />
              {isSelectedVariantsMode ? (
                <Button variant="text" color="warning" startIcon={<PictureAsPdfIcon />} onClick={handleCreatePdf}>
                  Create Report
                </Button>
              ) : (
                <Button variant="text" startIcon={<AddIcon />} onClick={() => setIsCreateNewDialogOpen(true)}>
                  Create a new table
                </Button>
              )}
            </>
          )}
        </Stack>,
        document.getElementById('custom-toolbar-container'),
      )}
      <Box p={0} pb={0} spacing={3}>
        {/* <Stack direction="row">
          <SampleSelectorView />
          <Divider orientation="vertical" flexItem /> 
          <TrioView />
        </Stack> */}
        <Box sx={{ width: '100%' }}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs
              value={value}
              onChange={handleTabChange}
              aria-label="basic tabs example"
              variant="scrollable"
              scrollButtons="auto"
            >
              {sampleName &&
                tables.map((table, index) => (
                  <VarTab
                    icon={<CloseButton tableId={table.id} index={index} />}
                    label={table.title}
                    {...a11yProps({ index })}
                  />
                ))}
            </Tabs>
            {sampleName &&
              tables.map((table, index) => (
                <TabPanel value={value} index={index}>
                  <Stack key={table.id}>
                    <VariantsTable
                      id={table.id}
                      variant={table.variant}
                      //defaultTitle={table.title}
                      readonlyTitle={index === 0}
                      hpo={table.hpo}
                      predefinedFilter={table.predefinedFilter}
                      page={page}
                      pageSize={pageSize}
                      sortOrder={sortOrder}
                      setPage={setPage}
                      setPageSize={setPageSize}
                      setSortOrder={setSortOrder}
                      data={data}
                      status={status}
                    />
                  </Stack>
                </TabPanel>
              ))}
          </Box>
        </Box>
      </Box>
    </Page>
  )
}

const VariantsViewContainer = function (props) {
  const [store, setStore] = useState()

  // FIXME: Maybe there is a better way?
  useEffect(() => {
    if (!store) {
      setStore(createStore(variantFiltersReducer))
    }
  }, [store])

  if (!store) {
    return null
  }

  return (
    <Provider store={store}>
      <VariantsView {...props} />
    </Provider>
  )
}

VariantsViewContainer.propTypes = {
  ...VariantsView.propTypes,
}

export default VariantsViewContainer
