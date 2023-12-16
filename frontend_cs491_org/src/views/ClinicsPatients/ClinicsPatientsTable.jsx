import {
    Box,
    CircularProgress,
    Stack,
    Button,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    Divider,
    Chip,
    IconButton,
    DialogContentText,
    Typography,
    TextField,
  } from '@material-ui/core'
  import { makeStyles } from '@material-ui/styles'
  import { ArrowForward, Info, Note } from '@material-ui/icons'
  import MUIDataTable from 'mui-datatables'
  import React, { useState } from 'react'
  import { useNavigate } from 'react-router-dom'
  import DeleteIcon from '@material-ui/icons/Delete'
  import { fDateTime } from 'src/utils/formatTime'
  import JobStateStatus from '../common/JobStateStatus'
  import { deleteVcfFile } from '../../api/vcf'
  import { deleteFastqFile } from '../../api/fastq'
  import { useFiles, annotateFile, useBedFiles, updateFinishInfo, updateFileNotes } from '../../api/file'
  
  import ExpandOnClick from 'src/components/ExpandOnClick'
  import AnalysedCheckbox from '../common/AnalysedCheckbox'
  
  import VariantDasboard2 from '../common/VariantDasboard2'
  
  const EditableNote = ({ note, onSave, details }) => {
    const [isEditing, setIsEditing] = useState(false)
    const [currentNote, setCurrentNote] = useState(note)
  
    const handleEdit = () => {
      setIsEditing(true)
    }
  
    const handleSave = () => {
      onSave(currentNote)
      setIsEditing(false)
    }
  
    const handleChange = (e) => {
      setCurrentNote(e.target.value)
    }
  
    return (
      <Box p={2}>
        {isEditing ? (
          <Box>
            <TextField multiline value={currentNote} onChange={handleChange} />
            <Box p={1} />
            <Button variant="contained" onClick={handleSave}>
              Save
            </Button>
          </Box>
        ) : (
          <Box>
            <Typography>{currentNote}</Typography>
            <Divider />
            {details.date && details.person && (
              <Typography variant="caption">
                {details.date && fDateTime(details.date)} {details.person ? `by ${details.person}` : null}
              </Typography>
            )}
            <Box p={1} />
            <Button variant="contained" onClick={handleEdit}>
              Edit
            </Button>
          </Box>
        )}
      </Box>
    )
  }
  
  const getStatusLabel = (row) => {
    const analyzeStatus = row?.analyses?.status ? row.analyses.status : null
    const annotationStatus = row?.annotations?.status ? row.annotations.status : null
    const is_annotated = row?.is_annotated
    if (!is_annotated) return 'WAITING'
    // if we're finished with annotating, return analysis status
    if (!annotationStatus || annotationStatus === 'DONE') return analyzeStatus
    return 'ANNO_' + annotationStatus
  }
  
  const useStyles = makeStyles((theme) => ({
    expandedCell: {
      boxShadow: theme.shadows[3]
        .split('),')
        .map((s) => `inset ${s}`)
        .join('),'),
    },
  }))
  
  const DeleteFileButton = ({ onClickConfirm }) => {
    const [dialogOpen, setDialogOpen] = useState(false)
  
    const handleClickDelete = () => {
      setDialogOpen(true)
    }
  
    const handleCloseDialog = () => {
      setDialogOpen(false)
    }
  
    const handleClickConfirm = () => {
      setDialogOpen(false)
      onClickConfirm()
    }
  
    return (
      <>
        <IconButton sx={{ '&:hover': { color: 'error.main' } }} onClick={handleClickDelete}>
          <DeleteIcon />
        </IconButton>
        <Dialog open={dialogOpen} onClose={handleCloseDialog}>
          <DialogTitle>Danger: Delete file</DialogTitle>
          <Box p={0.5} />
          <DialogContent>
            <DialogContentText>Do you really want to delete this file?</DialogContentText>
            <DialogContentText>This will delete the file and all associated analyses.</DialogContentText>
            <DialogContentText>
              <strong>This action cannot be undone.</strong>
            </DialogContentText>
          </DialogContent>
          <DialogActions>
            <Button onClick={handleCloseDialog}>Cancel</Button>
            <Button onClick={handleClickConfirm} autoFocus color="error">
              Delete
            </Button>
          </DialogActions>
        </Dialog>
      </>
    )
  }
  
  const GoToSampleDashboard = function ({ fileId, sampleName }) {
    const navigate = useNavigate()
  
    const handleClick = () => {
      navigate(`/libra/sample/${fileId}/${sampleName}`)
    }
  
    return (
      <Button variant="contained" onClick={handleClick} size="small">
        <ArrowForward />
      </Button>
    )
  }
  
  const SamplesView = function () {
    //const classes = useStyles()
    const filesApi = useFiles()
    const bedFilesApi = useBedFiles()
    const { status, data = [] } = filesApi.query
    const { data: bedFiles = [] } = bedFilesApi.query
    const [isAnnotationModalOpen, setAnnotationModalOpen] = useState(false)
    const [selectedFile, setSelectedFile] = useState(null)
  
    const setFinishedInfo = (row) => {
      const id = row.vcf_id ? row.vcf_id : row.fastq_pair_id
      updateFinishInfo(id).then(() => {
        filesApi.refresh()
      })
    }
  
    const setFileNotes = (row, notes) => {
      const id = row.vcf_id ? row.vcf_id : row.fastq_pair_id
      updateFileNotes(id, notes).then(() => {
        filesApi.refresh()
      })
    }
  
    const handleFileAnnotation = (annotation) => {
      const { id, type } = selectedFile?.vcf_id
        ? { id: selectedFile.vcf_id, type: 'VCF' }
        : { id: selectedFile.fastq_pair_id, type: 'FASTQ' }
      annotateFile(id, annotation, type).then((res) => {
        filesApi.refresh()
        setAnnotationModalOpen(false)
      })
    }
      const handleButtonChange = () => {
        //do the change here
        const { id, type } = selectedFile?.vcf_id
        ? { id: selectedFile.vcf_id, type: 'VCF' }
        : { id: selectedFile.fastq_pair_id, type: 'FASTQ' }
        let annotation = {
            type: 'Exome',
            machine: 'Illumina',
            kit: 0,
            panel: '',
            germline: true,
            alignment: 'BWA',
            reference: 'GRCh38',
            isSnp: false,
            isCnv: false,
            cnvAnalysis: 'xhmm+decont',
        }
        annotateFile(id, annotation, type).then((res) => {
            filesApi.refresh()
            //setAnnotationModalOpen(false)
        })
    }
  
    const handleAnnotationModelOpen = (row) => {
      setSelectedFile(row)
      setAnnotationModalOpen(true)
    }
  
    const COLUMNS = [
      {
        name: 'delete',
        label: 'Delete',
        options: {
          filter: false,
          sort: false,
          customBodyRenderLite(dataIndex) {
            const row = data[dataIndex]
            if (!row) return null
  
            const handleClickConfirm = () => {
              if (row.vcf_id) {
                deleteVcfFile(row.vcf_id)
              } else if (row.fastq_pair_id) {
                deleteFastqFile(row.fastq_file_1.fastq_file_id)
                deleteFastqFile(row.fastq_file_2.fastq_file_id)
              } else {
                deleteFastqFile(row.fastq_file_id)
              }
  
              filesApi.refresh()
            }
  
            return <DeleteFileButton onClickConfirm={handleClickConfirm} />
          },
        },
      },
      {
        name: 'created_at',
        label: 'Uploaded At',
        options: {
          filter: false,
          sort: true,
          customBodyRenderLite(dataIndex) {
            const row = data[dataIndex]
            return row ? fDateTime(row.created_at) : null
          },
        },
      },
      {
        name: 'finished_at',
        label: 'Completed',
        options: {
          filter: true,
          sort: false,
          customBodyRenderLite(dataIndex) {
            const row = data[dataIndex]
            return row ? (
              <AnalysedCheckbox
                checked={row.is_finished}
                onChange={(e) => setFinishedInfo(row)}
                details={{ date: row.finish_time, person: row.finish_person }}
              />
            ) : null
          },
        },
      },
      {
        name: 'notes',
        label: 'Notes',
        options: {
          filter: false,
          sort: false,
          customBodyRenderLite(dataIndex) {
            const row = data[dataIndex]
            return row ? (
              <ExpandOnClick
                expanded={
                  <EditableNote
                    note={row.file_notes}
                    onSave={(notes) => setFileNotes(row, notes)}
                    details={{ date: row.notes_time, person: row.notes_person }}
                  />
                }
              >
                {({ ref, onClick }) => (
                  <IconButton variant="contained" ref={ref} onClick={onClick}>
                    <Note />
                  </IconButton>
                )}
              </ExpandOnClick>
            ) : null
          },
        },
      },
      {
        name: 'name',
        label: 'Filename',
        options: {
          filter: true,
          sort: true,
          customBodyRenderLite: (dataIndex) => {
            const row = data[dataIndex]
            if (!row) return null
            if (row?.fastq_pair_id) {
              return (
                <Stack direction="row" spacing={1}>
                  <Chip label={row.fastq_file_1.name} />
                  <Chip label={row.fastq_file_2.name} />
                </Stack>
              )
            }
            return <Chip label={row.name} />
          },
        },
      },
      {
        name: 'sample_name',
        label: 'Sample',
        options: {
          filter: true,
          sort: true,
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
            const status = getStatusLabel(row)
            return status ? <JobStateStatus status={status} /> : null
          },
        },
      },
      {
        name: 'go',
        label: 'Go',
        options: {
          filter: false,
          sort: false,
          customBodyRenderLite: (dataIndex) => {
            const row = data[dataIndex]
            const status = getStatusLabel(row)
            if (status === 'ANNO_RUNNING' || status === 'ANNO_PENDING') return null
            if (status.includes('ANNO') || status === 'WAITING')
              return (
                <Button variant="contained" color="info" onClick={() => handleAnnotationModelOpen(row)} size="small">
                  <Info />
                </Button>
              )
            return (
              <GoToSampleDashboard fileId={row.vcf_id ? row.vcf_id : row.fastq_pair_id} sampleName={row.sample_name} />
            )
          },
        },
      },
    ]
  
    switch (status) {
      case 'success':
        return (
          <>
            <VariantDasboard2
            open={isAnnotationModalOpen}
            handleButtonChange = {handleButtonChange}
            onClose={() => setAnnotationModalOpen()}
            />
            <MUIDataTable
              title="Files"
              data={data}
              columns={COLUMNS}
              options={{
                selectableRows: 'none',
                sortOrder: { name: 'created_at', direction: 'desc' },
                expandableRows: false,
                print: false,
                viewColumns: true,
                download: false,
              }}
            />
          </>
        )
      default:
        return <CircularProgress />
    }
  }
  
  export default SamplesView
  