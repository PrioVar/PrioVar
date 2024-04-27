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
  import { ArrowForward, Info, Note, Add } from '@material-ui/icons'
  import MUIDataTable from 'mui-datatables'
  import React, { useState, useEffect } from 'react'
  import { useNavigate } from 'react-router-dom'
  import DeleteIcon from '@material-ui/icons/Delete'
  import { fDateTime } from 'src/utils/formatTime'
  import JobStateStatus from '../common/JobStateStatus'
  import { deleteVcfFile } from '../../api/vcf'
  //import { useFiles, annotateFile, useBedFiles, updateFinishInfo, updateFileNotes, fetchClinicianPatients, fetchCurrentClinicianName } from '../../api/file'
  import { useFiles, annotateFile, updateFinishInfo, updateFileNotes, fetchClinicianPatients, fetchCurrentClinicianName } from '../../api/file'
  import { PATH_DASHBOARD, } from '../../routes/paths'
  import { Link as RouterLink } from 'react-router-dom'
  import ExpandOnClick from 'src/components/ExpandOnClick'
  import AnalysedCheckbox from '../common/AnalysedCheckbox'
  import axios from '../../utils/axios'
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
  
  const GoToSampleDashboard = function ({ fileName }) {
    const navigate = useNavigate()
    console.log(fileName)
  
    const handleClick = () => {
      navigate(`/libra/sample/${fileName}`)
    }
  
    return (
      <Button variant="contained" onClick={handleClick} size="small">
        <ArrowForward />
      </Button>
    )
  }

  const MyPatientsTable = function () {
    //const classes = useStyles()
    let navigate = useNavigate()
    const filesApi = useFiles()
    //const bedFilesApi = useBedFiles()
    const [data, setData] = useState([])
    const [isLoading, setIsLoading] = useState(true)
    const [clinicianName, setClinicianName] = useState('');
    //const { status, data = [] } = filesApi.query
    //const { data: bedFiles = [] } = bedFilesApi.query
    const [isAnnotationModalOpen, setAnnotationModalOpen] = useState(false)
    const [selectedFile, setSelectedFile] = useState(null)
  
    const fetchClinicianName = async () => {
      /*
      const clinicianId = localStorage.getItem('clinicianId');
      if (clinicianId) {
      */
      try {
          const response = await fetchCurrentClinicianName();
          setClinicianName(response.data); // Assuming the response contains the clinician object with a name attribute

        } catch (error) {
          console.error('Failed to fetch clinician data:', error);
      }
    };

    const fetchAllPatients = async () => {
      setIsLoading(true);
      try {
        const data = await fetchClinicianPatients()
        setData(data)
        return data;
    
      } catch (error) {
        console.error(error);
        throw error;
      }
      finally {
        setIsLoading(false);
      }
    };

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

    const handleSeeSimilarPatients = (row) => {
        navigate(PATH_DASHBOARD.general.similarPatients, { state: { detail: row } });
    }

    const handleAddPatient = (row) => {}

    const handleDetails = (row) => {
        // TODO
    }
    
    useEffect(() => {
      fetchClinicianName();  
      fetchAllPatients();
    }, [])

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
              deleteVcfFile(row.patientId).then(() => {
              });
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
            return row && row.file ? fDateTime(row.file.createdAt) : null;

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
                checked={row.file?.finishedAt != null}
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
            const row = data[dataIndex];
            return row && row.file ? (
              <ExpandOnClick
                expanded={
                  <EditableNote
                    note={row.file.clinicianComments}
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
            ) : null;
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
            if (!row.file) return null
            return <Chip label={row.file.fileName} />
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
            const status = getStatusLabel(row)
            return status ? <JobStateStatus status={status} /> : null
          },
        },
      },
      {
        name: 'similar_patients',
        label: 'See similar patients',
        options: {
          filter: false,
          sort: true,
          customBodyRenderLite(dataIndex) {
            const row = data[dataIndex]
            return (
                <Button variant="contained" color="info" onClick={() => handleSeeSimilarPatients(row)} size="small">
                  <Info />
                </Button>
              )
          },
        },
      },
      {
        name: 'details',
        label: 'Details',
        options: {
          filter: false,
          sort: true,
          customBodyRenderLite(dataIndex) {
            const row = data[dataIndex]
            return (
                <Button variant="contained" color="info" onClick={() => handleDetails(row)} 
                component={RouterLink} to={PATH_DASHBOARD.general.patientDetails} size="small">
                  <Info />
                </Button>
              )
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
            if(!row.file) return null
            if (status === 'ANNO_RUNNING' || status === 'ANNO_PENDING') return null
            if (status.includes('ANNO') || status === 'WAITING')
              return (
                <GoToSampleDashboard fileName={row.file.fileName} />
              )
            return (
              <GoToSampleDashboard fileName={row.file.fileName} />
            )
          },
        },
      },
    ]
    return (
      <>
        {isLoading ? 
          (<CircularProgress/>) : 
          (
            <>
            <Box display="flex" justifyContent="flex-end" mt={2}> 
            <Button 
                variant="contained" 
                color="info" 
                component={RouterLink} to={PATH_DASHBOARD.general.files}
                size="small"
            >
                <Add /> 
                Add Patient 
            </Button>
            </Box>
            <VariantDasboard2
            open={isAnnotationModalOpen}
            handleButtonChange = {handleButtonChange}
            onClose={() => setAnnotationModalOpen()}
            />
            <MUIDataTable
              title={`All patients of clinician ${clinicianName || '...'}`}
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
          ) }
      </>

    )
  }
  
  export default MyPatientsTable
  
/*
switch (status) {
        case 'success':
          return (
            <>
              <Box display="flex" justifyContent="flex-end" mt={2}> 
              <Button 
                  variant="contained" 
                  color="info" 
                  component={RouterLink} to={PATH_DASHBOARD.general.files}
                  size="small"
              >
                  <Add /> 
                  Add Patient 
              </Button>
              </Box>
              <VariantDasboard2
              open={isAnnotationModalOpen}
              handleButtonChange = {handleButtonChange}
              onClose={() => setAnnotationModalOpen()}
              />
              <MUIDataTable
                title="All patients of clinic ABC"
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
*/