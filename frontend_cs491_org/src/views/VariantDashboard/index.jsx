// material-ui
import {
  Box,
  Button,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Grid,
  Stack,
  Tab,
  Tabs,
  Tooltip,
  Typography,
} from '@material-ui/core'
import { withStyles } from '@material-ui/styles'
// hooks
import { useState, useMemo, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { useFiles } from 'src/api/file/list'
import ReactDOM from 'react-dom'

// constants
import { HPO_OPTIONS, DASHBOARD_CONFIG } from 'src/constants'

// components
import AnalysesTable from './AnalysesTable'
import Tags from 'src/components/Tags'
import Page from 'src/components/Page'
import NavbarRoutes from '../VariantsView/NavbarRoutes'

// api utils
import { useHpo } from '../../api/vcf'
import { startJobsForVcfFile } from '../../api/vcf/job'
import { startJobsForFastqFile } from '../../api/fastq'

const Loading = function () {
  return (
    <Stack direction="row" justifyContent="center">
      <CircularProgress size="10vh" />
    </Stack>
  )
}


const CreateAnalysisDialog = function ({ open, onClose, onClickCreateAnalysis, fileType, title }) {
  const [snpChecked, setSnpChecked] = useState(false)
  const [cnvChecked, setCnvChecked] = useState(false)
  const [alignmentChecked, setAlignmentChecked] = useState(false)
  const [cnvAnalysis, setCnvAnalysis] = useState('xhmm+decont')

  const handleClickCreateAnalysis = () => {
    onClickCreateAnalysis({
      cnv: cnvChecked ? cnvAnalysis : null,
      snp: snpChecked ? 'gatk' : null,
      alignment: alignmentChecked ? 'bwa' : null,
    })
  }

  return (
    <Dialog open={open} onClose={onClose} maxWidth={false} PaperProps={{ sx: { width: 600 } }}>
      {title && (
        <>
          <DialogTitle>Dou you want to start the analysis of {title}?</DialogTitle>
          <DialogContent>

          </DialogContent>
          <DialogActions>
            <Button onClick={onClose} color="secondary">
              Cancel
            </Button>
            <Tooltip title={alignmentChecked || fileType === 'VCF' ? '' : 'FASTQ files require alignment'}>
              <Box>
                <Button
                  onClick={handleClickCreateAnalysis}
                  color="primary"
                  disabled={!alignmentChecked && fileType !== 'VCF'}
                >
                  Start
                </Button>
              </Box>
            </Tooltip>
          </DialogActions>
        </>
      )}
    </Dialog>
  )
}

const TABS = ['Analysis']

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

const VariantDasboard = () => {
  const { fileId, sampleName } = useParams()

  useEffect(() => {
    if (localStorage.getItem('dashboardSampleID') !== fileId) {
      localStorage.setItem('dashboardSampleID', fileId)
    }
  }, [fileId])

  const filesApi = useFiles()
  const { status, data = [] } = filesApi.query
  const fileDetails = useMemo(
    () => data.find((f) => f.vcf_id === fileId || f.fastq_pair_id === fileId),
    [data, fileId, filesApi],
  )

  const [isAnalysisOpen, setAnalysisOpen] = useState(false)
  const [tab, setTab] = useState(0)
  const handleTabChange = (event, newValue) => {
    setTab(newValue)
  }

  /*
  const [gender, setGender] = useState(fileDetails?.details?.sex)
  const [startAge, setStartAge] = useState(fileDetails?.details?.symptoms_start_age)
  const [age, setAge] = useState(fileDetails?.details?.age)
  const [is_inbred, setIsInbred] = useState(fileDetails?.details?.is_inbred)
  const [is_progressing, setProgressing] = useState(fileDetails?.details?.is_progressing)
  const [notes, setNotes] = useState(fileDetails?.details?.notes)
  */
  //const handleGenderChange = (e) => setGender(e.target.value)
  //const handleStartAgeChange = (e) => setStartAge(e.target.value)
  //const handleAgeChange = (e) => setAge(e.target.value)
  /*  const handleIsInbredChange = (e) => setIsInbred(e.target.checked)
  const handleIsProgressingChange = (e) => setProgressing(e.target.checked) */
  //const handleSetNotes = (e) => setNotes(e.target.value)

  const handleCreateAnalysis = async ({ cnv, snp, alignment }) => {
    const activeFileCopy = { ...fileDetails }
    setAnalysisOpen(false)
    if (activeFileCopy.vcf_id) {
      await startJobsForVcfFile(activeFileCopy.vcf_id)
    } else if (activeFileCopy.fastq_pair_id) {
      await startJobsForFastqFile(activeFileCopy.fastq_pair_id, { cnv, snp, alignment })
    }
    await filesApi.refresh()
  }

  return (
    <>
      <CreateAnalysisDialog
        open={isAnalysisOpen}
        onClose={() => setAnalysisOpen(null)}
        onClickCreateAnalysis={handleCreateAnalysis}
        fileType={fileDetails?.vcf_id ? 'VCF' : 'FASTQ'}
        title={fileDetails?.sample_name}
        key={'create-analysis-dialog' + fileId}
      />
      <Page title="Variant Dashboard | PrioVar">
        {ReactDOM.createPortal(
          <Stack direction="row" spacing={2}>
            <NavbarRoutes navConfig={DASHBOARD_CONFIG} />
          </Stack>,
          document.getElementById('custom-toolbar-container'),
        )}
        {fileDetails ? (
          <Box flexDirection="column" spacing={3} mt={2}>
            <Typography textAlign="center" variant="h4">
              Sample: {sampleName} | File: {fileDetails?.vcf_id ? 'VCF' : 'FASTQ'}
            </Typography>
            <Tabs
              value={tab}
              onChange={handleTabChange}
              aria-label="basic tabs example"
              variant="fullWidth"
              scrollButtons="auto"
            >
              {sampleName && TABS.map((table) => <VarTab label={table} />)}
            </Tabs>
            
              <Stack key={`dashboard-tabPanel-${2}`}>
                <Grid item xs={12}>
                  {fileDetails && (
                    <AnalysesTable
                      fileId={fileDetails?.vcf_id || fileDetails?.fastq_pair_id || fileDetails?.fastq_file_id}
                      sampleName={fileDetails.sample_name}
                      data={fileDetails?.analyses?.list ?? []}
                      onCreate={() => setAnalysisOpen(true)}
                    />
                  )}
                </Grid>
              </Stack>
          </Box>
        ) : (
          <Loading />
        )}
      </Page>
    </>
  )
}

export default VariantDasboard

/*
const createTrioOptions = (rows) => {
  return rows.map((row) => ({
    label: row.sample_name,
    value: { fileId: row.id, sample: row.sample_name },
  }))
}
const ManageHpo = function ({ fileId }) {
  const [hpoList, setHpoList] = useHpo({ fileId })

  return <Tags title="Symptoms" options={HPO_OPTIONS} value={hpoList} onChange={setHpoList} />
}
const TrioSelector = function ({ options, trio = {}, sampleName, fileId }) {
  const [motherOption, setMotherOption] = useState({
    label: trio?.mother_sample_name,
    value: { fileId: trio?.mother_file, sample: trio?.mother_sample_name },
  })
  const [fatherOption, setFatherOption] = useState({
    label: trio?.father_sample_name,
    value: { fileId: trio?.father_file, sample: trio?.father_sample_name },
  })

  useLazyEffect(() => {
    updateTrio(fileId, { mother: motherOption.value, father: fatherOption.value })
  }, [fileId, motherOption, fatherOption])

  return (
    <Grid container direction="row" spacing={1}>
      <Grid item xs={6}>
        <Autocomplete
          options={options?.filter(
            (o) =>
              ![trio?.mother_sample_name, trio?.father_sample_name, sampleName, fatherOption.value.sample].includes(
                o.value.sample,
              ),
          )}
          renderInput={(params) => <TextField {...params} label="Mother" variant="outlined" />}
          value={motherOption}
          getOptionLabel={(option) => option.label || ''}
          onChange={(_e, newMother) => {
            newMother = newMother || {
              label: '',
              value: { fileId: undefined, sample: undefined },
            }

            setMotherOption(newMother)
          }}
        />
      </Grid>
      <Grid item xs={6}>
        <Autocomplete
          options={options.filter(
            (o) =>
              ![trio?.mother_sample_name, trio?.father_sample_name, sampleName, motherOption.value.sample].includes(
                o.value.sample,
              ),
          )}
          renderInput={(params) => <TextField {...params} label="Father" variant="outlined" />}
          value={fatherOption}
          getOptionLabel={(option) => option.label || ''}
          onChange={(_e, newFather) => {
            newFather = newFather || {
              label: '',
              value: { fileId: undefined, sample: undefined },
            }

            setFatherOption(newFather)
          }}
        />
      </Grid>
    </Grid>
  )
}
*/

/*
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
*/
/*

const handleSave = () => {
    const details = {
      gender,
      start_age: Number(startAge),
      age: Number(age),
      is_inbred,
      is_progressing,
      notes,
    }
    const { id, type } = fileDetails.vcf_id
      ? { id: fileDetails.vcf_id, type: 'VCF' }
      : { id: fileDetails.fastq_pair_id, type: 'FASTQ' }
    updateDetails(id, details, type)
    filesApi.refresh()
  }

  const trioOptions = createTrioOptions(data)

  */