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
  import { useState, useMemo, useEffect } from 'react'
  import { useParams } from 'react-router-dom'
  import useLazyEffect from 'src/hooks/useLazyEffect'
  import { useFiles } from 'src/api/file/list'
  import { updateDetails } from 'src/api/file'
  
  // constants
  import { HPO_OPTIONS } from 'src/constants'
  
  // components
  import Tags from 'src/components/Tags'
  
  
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
  
  const ManageHpo = function ({ fileId }) {
    const [hpoList, setHpoList] = useHpo({ fileId })
  
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
  
    const handleGenderChange = (e) => setGender(e.target.value)
    const handleStartAgeChange = (e) => setStartAge(e.target.value)
    const handleAgeChange = (e) => setAge(e.target.value)
    /*  const handleIsInbredChange = (e) => setIsInbred(e.target.checked)
    const handleIsProgressingChange = (e) => setProgressing(e.target.checked) */
    const handleSetNotes = (e) => setNotes(e.target.value)
  
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
      } /*else if (activeFileCopy.fastq_pair_id) {
        await startJobsForFastqFile(activeFileCopy.fastq_pair_id, { cnv, snp, alignment })
      }*/
      await filesApi.refresh()
    }
  
    return (
        <Dialog open={open} onClose={onClose}>
      <>
          {/*ReactDOM.createPortal(
            <Stack direction="row" spacing={2}>
              <NavbarRoutes navConfig={DASHBOARD_CONFIG} />
            </Stack>,
            document.getElementById('custom-toolbar-container'),
          )*/}
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
                      <ManageHpo fileId={fileId} sampleName={sampleName} />
                    </CardContent>
                  </Grid>
                  <Grid item container direction="row" xs={12}>
                    <Grid item xs={6}>
                      <CardHeader title="Parents" titleTypographyProps={{ variant: 'subtitle1' }} />
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
                      <Grid item container direction="row" xs={12}>
                        <Grid container item xs={6} sx={{ flexGrow: 1 }}>
                          <Grid item xs={12}>
                            <CardContent>
                              {fileDetails && (
                                <TrioSelector
                                  trio={fileDetails?.trio ? fileDetails.trio : null}
                                  options={trioOptions}
                                  samples={sampleName}
                                  fileId={fileId}
                                />
                              )}
                            </CardContent>
                          </Grid>
                          <Grid item xs={12} sx={{ flexGrow: 1 }}>
                            <CardHeader
                              title="Details"
                              titleTypographyProps={{ variant: 'subtitle1' }}
                              sx={{ paddingTop: 0 }}
                            />
                          </Grid>
                          <Grid container direction={'row'} item xs={12} height={'auto'}>
                            <Grid container item xs={6} direction={'row'}>
                              <Grid item xs={12}>
                                <CardContent>
                                  <FormControl fullWidth>
                                    <InputLabel id="select-age">Age</InputLabel>
                                    <Input type="number" value={age} onChange={handleAgeChange}></Input>
                                  </FormControl>
                                </CardContent>
                              </Grid>
                              <Grid item xs={12}>
                                <CardContent>
                                  <FormControl fullWidth>
                                    <InputLabel id="select-start-age">Symptoms start age</InputLabel>
                                    <Input type="number" value={startAge} onChange={handleStartAgeChange}></Input>
                                  </FormControl>
                                </CardContent>
                              </Grid>
                            </Grid>
                            <Grid container item xs={6} direction={'row'}>
                              <Grid xs={12} item>
                                <CardContent>
                                  <FormControl fullWidth>
                                    <InputLabel id="select-gender">Gender</InputLabel>
                                    <Select
                                      labelId="select-gender"
                                      value={gender}
                                      onChange={handleGenderChange}
                                      label="Gender"
                                    >
                                      <MenuItem value={'M'}>Male</MenuItem>
                                      <MenuItem value={'F'}>Female</MenuItem>
                                    </Select>
                                  </FormControl>
                                </CardContent>
                              </Grid>
                              {/*                     <Grid xs={12} item container direction="row" alignItems="center" justifyContent="space-between">
                        <InputLabel htmlFor="select-is-progressin">Is symptoms progressing?</InputLabel>
                        <Checkbox
                          id="select-is-progressin"
                          checked={is_progressing}
                          onClick={handleIsProgressingChange}
                        ></Checkbox>
                      </Grid>
                      <Grid xs={12} item container direction="row" alignItems="center" justifyContent="space-between">
                        <InputLabel htmlFor="select-is-inbred">Is there any inbred?</InputLabel>
                        <Checkbox id="select-is-inbred" checked={is_inbred} onClick={handleIsInbredChange}></Checkbox>
                      </Grid> */}
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
  