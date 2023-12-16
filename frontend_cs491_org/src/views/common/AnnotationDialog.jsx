import {
  Box,
  Button,
  CardHeader,
  CardContent,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Grid,
  MenuItem,
  TextField,
  Checkbox,
  Stack,
  Tooltip,
} from '@material-ui/core'
import { UploadMultiFile } from 'src/components/upload'
import { useState, useEffect } from 'react'

import axios from 'axios'
import { uploadFile } from '../../api/bed'

let cancelTokenSource = null

const AnnotationDialog = function ({ open, onClose, onClickAnnotate, fileType, title, bedFiles, bedFilesApi }) {
  const [type, setType] = useState('Exome')
  const [machine, setMachine] = useState('Illumina')
  const [kit, setKit] = useState(0)
  const [panel, setPanel] = useState('')
  const [germline, setGermline] = useState(true)
  const [alignment, setAlignment] = useState('BWA')
  const [reference, setReference] = useState('GRCh38')
  const [snpChecked, setSnpChecked] = useState(false)
  const [cnvChecked, setCnvChecked] = useState(false)
  const [cnvAnalysis, setCnvAnalysis] = useState('xhmm+decont')

  const [bedUploadOpen, setBedUploadOpen] = useState(false)
  const [fileQueue, setFileQueue] = useState([])
  const [state, setState] = useState('IDLE')
  const [uploadProgress, setUploadProgress] = useState(0)

  const [fileNames, setFileNames] = useState([])
  const [references, setReferences] = useState([])

  const handleDropMultiFile = async (filesToUpload) => {
    setFileQueue(filesToUpload)
    setFileNames(filesToUpload.map((file) => file.name))
    setReferences(filesToUpload.map(() => 'GRCh38'))
    setState('READY_TO_UPLOAD')

    // await uploadFile(fileToUpload)
    // await filesApi.refresh()
  }

  const handleUploadFiles = async (_filesToUpload) => {
    setState('UPLOADING')
  }

  const handleRemoveAll = () => {
    setFileQueue([])
  }

  const handleRemove = (file) => {
    setFileQueue(fileQueue.filter((f) => f !== file))
  }

  const handleCancel = () => {
    setFileQueue([])
    cancelTokenSource.cancel('Cancelled by user')
  }

  const NamingComponent = ({ file, index, fileNames, setFileNames, references, setReferences }) => {
    const handleFileNameChange = (value, index) => {
      const newFileNames = [...fileNames]
      newFileNames[index] = value
      setFileNames(newFileNames)
    }

    const handleReferenceChange = (value, index) => {
      const newReferences = [...references]
      newReferences[index] = value
      setReferences(newReferences)
    }
    return file ? (
      <Box>
        <CardContent>
          <Grid container spacing={2}>
            <Grid item xs={8}>
              <TextField
                key={`fname-${file.name}`}
                label="File name"
                fullWidth
                value={fileNames[index]}
                onChange={(e) => handleFileNameChange(e.target.value, index)}
              ></TextField>
            </Grid>
            <Grid item xs={4}>
              <TextField
                key={`fref-${file.name}`}
                label="Reference"
                select
                fullWidth
                value={references[index]}
                onChange={(e) => handleReferenceChange(e.target.value, index)}
              >
                <MenuItem value={'GRCh38'}>GRCh38</MenuItem>
                <MenuItem value={'GRCh37'}>GRCh37</MenuItem>
              </TextField>
            </Grid>
          </Grid>
        </CardContent>
      </Box>
    ) : null
  }

  useEffect(() => {
    const uploadFilesStateMachine = async () => {
      if (state === 'UPLOADING') {
        if (fileQueue.length > 0) {
          setUploadProgress(0)
          cancelTokenSource = axios.CancelToken.source()
          await uploadFile(
            fileQueue[0],
            (value) => setUploadProgress(value),
            cancelTokenSource.token,
            fileNames[0],
            references[0],
          )

          setFileQueue(fileQueue.slice(1))
          setFileNames(fileNames.slice(1))
          setReferences(references.slice(1))
          await bedFilesApi.refresh()
        } else {
          // Done uploading all the files
          setBedUploadOpen(false)
          setState('IDLE')
        }
      }
    }

    uploadFilesStateMachine()
  }, [state, fileQueue])

  const handleClickAnnotate = () => {
    onClickAnnotate({
      type,
      machine,
      kit,
      panel,
      germline,
      alignment,
      reference,
      isSnp: snpChecked,
      isCnv: cnvChecked,
      cnvAnalysis,
    })
  }

  const handleModalClose = () => {
    setBedUploadOpen(false)
    onClose()
  }

  return (
    <Dialog open={open} onClose={onClose}>
      {title && (
        <>
          <DialogTitle>Make annotation for {title}</DialogTitle>
          <DialogContent sx={{ paddingBottom: 0 }}>
            <CardHeader title="Sequence" titleTypographyProps={{ variant: 'subtitle1' }} />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <TextField
                    label="Machine"
                    defaultValue={0}
                    select
                    fullWidth
                    value={machine}
                    onChange={(e) => setMachine(e.target.value)}
                  >
                    <MenuItem value="Illumina">Illumina</MenuItem>
                    <MenuItem value="MGISEQ">MGISEQ</MenuItem>
                    <MenuItem value="BGISEQ">BGISEQ</MenuItem>
                  </TextField>
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Kit"
                    defaultValue={0}
                    select
                    fullWidth
                    value={kit}
                    onChange={(e) => setKit(e.target.value)}
                  >
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
                    <TextField
                      label="Capture Kit"
                      value={panel}
                      fullWidth
                      onChange={(e) => setPanel(e.target.value)}
                      select
                    >
                      {bedFiles.map((bedFile) => (
                        <MenuItem key={bedFile.id} value={bedFile.id}>
                          {bedFile.visible_name ? bedFile.visible_name : bedFile.original_name}
                        </MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                )}
                <Grid item xs={6}>
                  <TextField
                    label="Germline/Somatic"
                    defaultValue={0}
                    select
                    fullWidth
                    value={germline}
                    onChange={(e) => setGermline(e.target.value)}
                  >
                    <MenuItem value={germline}>Germline</MenuItem>
                    <MenuItem value={!germline}>Somatic</MenuItem>
                  </TextField>
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Alignment"
                    defaultValue={0}
                    select
                    fullWidth
                    value={alignment}
                    onChange={(e) => setAlignment(e.target.value)}
                  >
                    <MenuItem value={'BWA'}>BWA MEM v1</MenuItem>
                    <MenuItem value={'Bowtie'}>Bowtie</MenuItem>
                  </TextField>
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Reference Genome"
                    defaultValue={0}
                    select
                    fullWidth
                    value={reference}
                    onChange={(e) => setReference(e.target.value)}
                  >
                    <MenuItem value={'GRCh38'}>GRCh38</MenuItem>
                    <MenuItem value={'GRCh37'}>GRCh37</MenuItem>
                  </TextField>
                </Grid>
                <Grid item xs={12}>
                  <CardHeader
                    title="Starting Analysis"
                    titleTypographyProps={{ variant: 'subtitle1' }}
                    sx={{ paddingTop: '5px', paddingLeft: '0' }}
                  />
                </Grid>
                <Grid item xs={12}>
                  <Stack direction="row" spacing={1} alignItems="center">
                    <Box>
                      <Checkbox disableRipple checked={snpChecked} onChange={(e) => setSnpChecked(e.target.checked)} />
                    </Box>
                    <TextField label="SNP" defaultValue={0} select fullWidth disabled={!snpChecked}>
                      <MenuItem value={0}>GATK v4 HaplotypeCaller</MenuItem>
                      <MenuItem value={1}>Free-bayes</MenuItem>
                      <MenuItem value={2}>Mutect2</MenuItem>
                    </TextField>
                  </Stack>
                </Grid>
                <Grid item xs={12}>
                  <Stack direction="row" spacing={1} alignItems="center">
                    <Box>
                      <Checkbox disableRipple checked={cnvChecked} onChange={(e) => setCnvChecked(e.target.checked)} />
                    </Box>
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
            {bedUploadOpen && (
              <>
                <CardHeader
                  title="Upload New Capture Kit"
                  sx={{ paddingTop: 0 }}
                  titleTypographyProps={{ variant: 'subtitle1' }}
                />
                <CardContent sx={{ '&:last-child': { paddingBottom: 0 } }}>
                  {state === 'UPLOADING' && fileQueue.length > 0 && <Box></Box>}
                  <UploadMultiFile
                    minimal
                    files={fileQueue}
                    onDrop={handleDropMultiFile}
                    onRemove={handleRemove}
                    onRemoveAll={handleRemoveAll}
                    onUploadFiles={handleUploadFiles}
                    locked={state === 'UPLOADING'}
                    onCancel={handleCancel}
                    NamingComponent={NamingComponent}
                    fileNames={fileNames}
                    setFileNames={setFileNames}
                    references={references}
                    setReferences={setReferences}
                  />
                </CardContent>
              </>
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={handleModalClose} color="error">
              Cancel
            </Button>
            <Button onClick={() => setBedUploadOpen(true)} color="secondary">
              Upload New Capture Kit
            </Button>
            <Button onClick={handleClickAnnotate} color="primary">
              Annotate
            </Button>
          </DialogActions>
        </>
      )}
    </Dialog>
  )
}

export default AnnotationDialog
