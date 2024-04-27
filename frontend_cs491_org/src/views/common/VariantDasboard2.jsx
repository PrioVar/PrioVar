// material-ui
import {
    Alert,
    Autocomplete,
    Box,
    Button,
    CardHeader,
    CardContent,
    Checkbox,
    CircularProgress,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    Grid,
    MenuItem,
    Select,
    Stack,
    Tab,
    Tabs,
    TextField,
    Tooltip,
    Typography,
    InputLabel,
    FormControl,
    Input,
  } from '@material-ui/core'
  import { withStyles } from '@material-ui/styles'
  // hooks
  import React, { useState, useMemo, useEffect } from 'react'
  import { useParams } from 'react-router-dom'
  import useLazyEffect from 'src/hooks/useLazyEffect'
  import { useFiles } from 'src/api/file/list'
  import { updateDetails } from 'src/api/file'
  import ReactDOM from 'react-dom'
  
  // constants
  import { HPO_OPTIONS, DASHBOARD_CONFIG } from 'src/constants'
  
  // components
  import Tags from 'src/components/Tags'
  import Page from 'src/components/Page'
  import axios from 'axios';

  
  
  // api utils
  import { updateTrio, useHpo } from '../../api/vcf'
  import { startJobsForVcfFile } from '../../api/vcf/job'
  
  const Loading = function () {
    return (
      <Stack direction="row" justifyContent="center">
        <CircularProgress size="10vh" />
      </Stack>
    )
  }
  
  
  const createTrioOptions = (rows) => {
    return rows.map((row) => ({
      label: row.sample_name,
      value: { fileId: row.id, sample: row.sample_name },
    }))
  }
  
  const CreateAnalysisDialog = function ({ open, onClose, onClickCreateAnalysis, fileType, title }) {
    const [type, setType] = useState('Exome')
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
            <DialogTitle>Create a new analysis for {title}</DialogTitle>
            <DialogContent>
              <CardHeader title="Sequence" titleTypographyProps={{ variant: 'subtitle1' }} />
              <CardContent>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <TextField label="Machine" defaultValue={0} select fullWidth>
                      <MenuItem value={0}>Illumina</MenuItem>
                      <MenuItem value={1}>MGISEQ</MenuItem>
                      <MenuItem value={2}>BGISEQ</MenuItem>
                    </TextField>
                  </Grid>
                  <Grid item xs={6}>
                    <TextField label="Kit" defaultValue={0} select fullWidth>
                      <MenuItem value={0}>Agilent SureSelect Human All Exon V5 r2</MenuItem>
                      <MenuItem value={1}>Agilent SureSelect Human All Exon V5 UTR r2</MenuItem>
                      <MenuItem value={2}>Agilent SureSelect version 5</MenuItem>
                    </TextField>
                  </Grid>
  
                  <Grid item xs={6}>
                    <TextField label="Type" value={type} fullWidth onChange={(e) => setType(e.target.value)} select>
                      <MenuItem value="Exome">Whole Exome</MenuItem>
                      <MenuItem value="Genome">Whole Genome</MenuItem>
                      <MenuItem value="Gene Panel">Capture Kit</MenuItem>
                    </TextField>
                  </Grid>
                  {type === 'Gene Panel' && (
                    <Grid item xs={6}>
                      <TextField label="Panel" fullWidth />
                    </Grid>
                  )}
                  <Grid item xs={6}>
                    <TextField label="Germline/Somatic" defaultValue={0} select fullWidth>
                      <MenuItem value={0}>Germline</MenuItem>
                      <MenuItem value={1}>Somatic</MenuItem>
                    </TextField>
                  </Grid>
                </Grid>
              </CardContent>
              <CardHeader
                title={
                  <Stack direction="row" justifyContent="space-between">
                    <Typography variant="subtitle1">Analysis</Typography>
                    {fileType === 'VCF' && (
                      <Box>
                        <Alert severity="warning" sx={{ px: 1, py: 0 }}>
                          VCF detected
                        </Alert>
                      </Box>
                    )}
                  </Stack>
                }
              />
              <CardContent>
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <Stack direction="row" spacing={1} alignItems="center">
                      <Checkbox
                        disableRipple
                        disabled={fileType === 'VCF'}
                        checked={alignmentChecked}
                        onChange={(e) => {
                          setAlignmentChecked(e.target.checked)
                          if (!e.target.checked) {
                            setSnpChecked(false)
                            setCnvChecked(false)
                          }
                        }}
                      />
                      <TextField label="Alignment" defaultValue={0} select fullWidth disabled={!alignmentChecked}>
                        <MenuItem value={0}>BWA MEM v1</MenuItem>
                        {/* <MenuItem value={1}>Bowtie</MenuItem> */}
                      </TextField>
                    </Stack>
                  </Grid>
                  <Grid item xs={12}>
                    <Stack direction="row" spacing={1} alignItems="center">
                      <Tooltip
                        title={
                          alignmentChecked || fileType === 'VCF' ? '' : 'SNP analysis is not possible without alignment'
                        }
                      >
                        <Box>
                          <Checkbox
                            disableRipple
                            disabled={!alignmentChecked}
                            checked={snpChecked}
                            onChange={(e) => setSnpChecked(e.target.checked)}
                          />
                        </Box>
                      </Tooltip>
                      <TextField label="SNP" defaultValue={0} select fullWidth disabled={!snpChecked}>
                        <MenuItem value={0}>GATK v4 HaplotypeCaller</MenuItem>
                        <MenuItem value={1}>Free-bayes</MenuItem>
                        <MenuItem value={2}>Mutect2</MenuItem>
                      </TextField>
                    </Stack>
                  </Grid>
                  <Grid item xs={12}>
                    <Stack direction="row" spacing={1} alignItems="center">
                      <Tooltip
                        title={
                          alignmentChecked || fileType === 'VCF' ? '' : 'CNV analysis is not possible without alignment'
                        }
                      >
                        <Box>
                          <Checkbox
                            disableRipple
                            disabled={!alignmentChecked}
                            checked={cnvChecked}
                            onChange={(e) => setCnvChecked(e.target.checked)}
                          />
                        </Box>
                      </Tooltip>
                      <TextField
                        label="CNV"
                        value={cnvAnalysis}
                        onChange={(e) => setCnvAnalysis(e.target.value)}
                        select
                        fullWidth
                        disabled={!cnvChecked}
                      >
                        <MenuItem value={'xhmm+decont'}>XHMM + DECoNT</MenuItem>
                        <MenuItem value={'xhmm'}>XHMM</MenuItem>
                      </TextField>
                    </Stack>
                  </Grid>
                </Grid>
              </CardContent>
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
                    Create
                  </Button>
                </Box>
              </Tooltip>
            </DialogActions>
          </>
        )}
      </Dialog>
    )
  }
  
  const ManageHpo = function ({ fileId , hpoList, setHpoList}) {
  
    return <Tags title="Symptoms" options={HPO_OPTIONS} value={hpoList} onChange={setHpoList} />
  }
  
  const TABS = ['Sample Details']
  
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
  
  const VariantDasboard2 = function ({ open, onClose, handleButtonChange, vcfFileId }) {
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
  
    const [gender, setGender] = useState(fileDetails?.details?.sex)
    const [startAge, setStartAge] = useState(fileDetails?.details?.symptoms_start_age)
    const [age, setAge] = useState(fileDetails?.details?.age)
    const [is_inbred, setIsInbred] = useState(fileDetails?.details?.is_inbred)
    const [is_progressing, setProgressing] = useState(fileDetails?.details?.is_progressing)
    const [notes, setNotes] = useState(fileDetails?.details?.notes)
    const [name, setName] = useState(fileDetails?.details?.name)
    const [healthCenterId, setHealthCenterId] = useState(localStorage.getItem('healthCenterId') || '');
  
    const handleGenderChange = (e) => setGender(e.target.value)
    const handleAgeChange = (e) => setAge(e.target.value)
    const handleNameChange = (e) => setName(e.target.value)
    useEffect(() => {
        localStorage.setItem('healthCenterId', healthCenterId);
      }, [healthCenterId]);
    /*  const handleIsInbredChange = (e) => setIsInbred(e.target.checked)
    const handleIsProgressingChange = (e) => setProgressing(e.target.checked) */
    const handleSetNotes = (e) => setNotes(e.target.value)
  
    const handleSave = async () => {
      const details = {
        gender,
        start_age: Number(startAge),
        age: Number(age),
        is_inbred,
        is_progressing,
        notes,
      }
      const phenotypeTerms = hpoList.map(item => {
        // Extract the numeric part of the HP code and remove leading zeros
        const id = parseInt(item.value.split(':')[1], 10);
        return { id };
      });
      let request = {
        patient: {
            name: name,
            age: age,
            sex: gender,
            clinicalHistory: notes,
            medicalCenter: {
                id: localStorage.getItem('healthCenterId'),
            }
          },
        phenotypeTerms: phenotypeTerms,
        vcfFileId: vcfFileId,
        clinicianId: localStorage.getItem('clinicianId')
      }
      try {
        const response = await axios.post('http://localhost:8080/patient/addPatientWithPhenotype', request);
        console.log("SUCCESS!")

      } catch (error) {
        console.log("FAIL!")
        console.error('add patient error:', error.response);
      }
    }
  
    const [hpoList, setHpoList] = useHpo({ fileId })
    return (
        <Dialog open={open} onClose={onClose}>
            <Box flexDirection="column" spacing={3} mt={2}>

              <Typography textAlign="center" variant="h4">
                File: VCF
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
  
                <Stack key={`dashboard-tabPanel-${1}`}>
                  <Grid item sx={{ minWidth: '65%', flexGrow: 1 }}>
                    <CardHeader title="Symptoms (HPO)" titleTypographyProps={{ variant: 'subtitle1' }} />
                    <CardContent>
                      <ManageHpo fileId={fileId} sampleName={sampleName} hpoList={hpoList} setHpoList={setHpoList} />
                    </CardContent>
                  </Grid>
                  <Grid item container direction="row" xs={12}>
                    <Grid item xs={6}></Grid>
                    <Grid item xs={6}>
                      <CardHeader title="Clinical History"titleTypographyProps={{ variant: 'subtitle1' }} sx={{ paddingLeft: 0 }} />
                    </Grid>
                  </Grid>
                    <Grid item container xs={12}>
                      <Grid container item xs={6} sx={{ flexGrow: 1 }}>
                        <Grid item xs={12} sx={{ flexGrow: 1 }}>
                          <CardHeader title="Details" titleTypographyProps={{ variant: 'subtitle1' }} sx={{ paddingTop: 0 }} />
                        </Grid>
                          <Grid container item xs={6} >
                            <CardContent>
                              <FormControl fullWidth>
                                <InputLabel id="select-name">Name</InputLabel>
                                <Input type="text" value={name} onChange={handleNameChange}></Input>
                              </FormControl>
                            </CardContent>
                            <Grid item xs={12}>
                              <CardContent>
                                <FormControl fullWidth>
                                  <InputLabel id="select-age">Age</InputLabel>
                                  <Input type="number" value={age} onChange={handleAgeChange}></Input>
                                </FormControl>
                              </CardContent>
                            </Grid>
                          </Grid>
                          <Grid container item xs={6} direction={'row'}>
                            <Grid xs={12} item>
                              <CardContent>
                                <FormControl fullWidth>
                                  <InputLabel id="select-gender">Sex</InputLabel>
                                  <Select labelId="select-gender" value={gender} onChange={handleGenderChange} label="Gender" >
                                    <MenuItem value={'Male'}>Male</MenuItem>
                                    <MenuItem value={'Female'}>Female</MenuItem>
                                  </Select>
                                </FormControl>
                              </CardContent>
                            </Grid>
                          </Grid>
                      </Grid>
                      <Grid item xs={6}>
                        <CardContent sx={{ height: '100%', paddingLeft: 0 }}>
                          <TextField fullWidth id="details-notes" value={notes} onChange={handleSetNotes} multiline rows={11} ></TextField>
                        </CardContent>
                      </Grid>
                    </Grid>
                    <Grid item xs={12} sx={{ flexGrow: 1 }} alignSelf="end" mr={3}>
                      <DialogActions>
                      <Button color="primary" variant="contained" onClick={handleSave}>
                        Save Details
                      </Button>
                      </DialogActions>
                </Grid>
              </Stack>
            </Box>

      </Dialog>
    )
  }
  
  export default VariantDasboard2

/*
const VariantDasboard2 = function ({ open, onClose, handleButtonChange }) {
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
  
    const [gender, setGender] = useState(fileDetails?.details?.sex)
    const [startAge, setStartAge] = useState(fileDetails?.details?.symptoms_start_age)
    const [age, setAge] = useState(fileDetails?.details?.age)
    const [is_inbred, setIsInbred] = useState(fileDetails?.details?.is_inbred)
    const [is_progressing, setProgressing] = useState(fileDetails?.details?.is_progressing)
    const [notes, setNotes] = useState(fileDetails?.details?.notes)
    const [name, setName] = useState(fileDetails?.details?.name)
    const [healthCenterId, setHealthCenterId] = useState(localStorage.getItem('healthCenterId') || '');
  
    const handleGenderChange = (e) => setGender(e.target.value)
    const handleStartAgeChange = (e) => setStartAge(e.target.value)
    const handleAgeChange = (e) => setAge(e.target.value)
    const handleNameChange = (e) => setName(e.target.value)
    const handleHealthCenterChange = (e) => setHealthCenterId(e.target.value)
    React.useEffect(() => {
        localStorage.setItem('healthCenterId', healthCenterId);
      }, [healthCenterId]);
    const handleSetNotes = (e) => setNotes(e.target.value)
  
    const handleSave = async () => {
      const details = {
        gender,
        start_age: Number(startAge),
        age: Number(age),
        is_inbred,
        is_progressing,
        notes,
      }
      const phenotypeTerms = hpoList.map(item => {
        // Extract the numeric part of the HP code and remove leading zeros
        const id = parseInt(item.value.split(':')[1], 10);
        return { id };
      });
      let request = {
        patient: {
            name: name,
            age: age,
            sex: gender,
            clinicalHistory: notes,
            medicalCenter: {
                id: healthCenterId,
            }
          },
        phenotypeTerms: phenotypeTerms
      }
      console.log("request:")
      console.log(request)
      try {
        const response = await axios.post('http://localhost:8080/patient/addPatientWithPhenotype', request);
        console.log("SUCCESS!")

      } catch (error) {
        console.log("FAIL!")
        console.error('add patient error:', error.response);
      }
      const { id, type } = fileDetails.vcf_id
        ? { id: fileDetails.vcf_id, type: 'VCF' }
        : { id: fileDetails.fastq_pair_id, type: 'FASTQ' }
      updateDetails(id, details, type)
      {onClose()}
      {handleButtonChange()}
      //handle button change
      filesApi.refresh()
    }
  

    const trioOptions = createTrioOptions(data)
  
    const handleCreateAnalysis = async ({ cnv, snp, alignment }) => {
      const activeFileCopy = { ...fileDetails }
      setAnalysisOpen(false)
      if (activeFileCopy.vcf_id) {
        await startJobsForVcfFile(activeFileCopy.vcf_id)
      } 
      await filesApi.refresh()
    }
  
    const [hpoList, setHpoList] = useHpo({ fileId })
    return (
        <Dialog open={open} onClose={onClose}>
      <>
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
  
                <Stack key={`dashboard-tabPanel-${1}`}>
                  <Grid item sx={{ minWidth: '65%', flexGrow: 1 }}>
                    <CardHeader title="Symptoms (HPO)" titleTypographyProps={{ variant: 'subtitle1' }} />
                    <CardContent>
                      <ManageHpo fileId={fileId} sampleName={sampleName} hpoList={hpoList} setHpoList={setHpoList} />
                    </CardContent>
                  </Grid>
                  <Grid item container direction="row" xs={12}>
                    <Grid item xs={6}>
                    </Grid>
                    <Grid item xs={6}>
                      <CardHeader
                        title="Clinical History"
                        titleTypographyProps={{ variant: 'subtitle1' }}
                        sx={{ paddingLeft: 0 }}
                      />
                    </Grid>
                  </Grid>
                  {fileDetails?.details && (
                    <>
                      <Grid item container xs={12}>
                        <Grid container item xs={6} sx={{ flexGrow: 1 }}>
                          <Grid item xs={12} sx={{ flexGrow: 1 }}>
                            <CardHeader
                              title="Details"
                              titleTypographyProps={{ variant: 'subtitle1' }}
                              sx={{ paddingTop: 0 }}
                            />
                          </Grid>
                            <Grid container item xs={6} >
                             <CardContent>
                                  <FormControl fullWidth>
                                    <InputLabel id="select-name">Name</InputLabel>
                                    <Input type="text" value={name} onChange={handleNameChange}></Input>
                                  </FormControl>
                                </CardContent>
                              <Grid item xs={12}>
                                <CardContent>
                                  <FormControl fullWidth>
                                    <InputLabel id="select-age">Age</InputLabel>
                                    <Input type="number" value={age} onChange={handleAgeChange}></Input>
                                  </FormControl>
                                </CardContent>
                              </Grid>
                            </Grid>
                            <Grid container item xs={6} direction={'row'}>
                              <Grid xs={12} item>
                                <CardContent>
                                  <FormControl fullWidth>
                                    <InputLabel id="select-gender">Sex</InputLabel>
                                    <Select
                                      labelId="select-gender"
                                      value={gender}
                                      onChange={handleGenderChange}
                                      label="Gender"
                                    >
                                      <MenuItem value={'Male'}>Male</MenuItem>
                                      <MenuItem value={'Female'}>Female</MenuItem>
                                    </Select>
                                  </FormControl>
                                </CardContent>
                              </Grid>
                            </Grid>
                        </Grid>
                        <Grid item xs={6}>
                          <CardContent sx={{ height: '100%', paddingLeft: 0 }}>
                            <TextField
                              fullWidth
                              id="details-notes"
                              value={notes}
                              onChange={handleSetNotes}
                              multiline
                              rows={11}
                            ></TextField>
                          </CardContent>
                        </Grid>
                      </Grid>
                      <Grid item xs={12} sx={{ flexGrow: 1 }} alignSelf="end" mr={3}>
                        <DialogActions>
                        <Button color="primary" variant="contained" onClick={handleSave}>
                          Save Details
                        </Button>
                    </DialogActions>
                      </Grid>
                    </>
                  )}
                </Stack>
  
            </Box>
          ) : (
            <Loading />
          )}
      </>
      </Dialog>
    )
  }
  
  export default VariantDasboard2
*/