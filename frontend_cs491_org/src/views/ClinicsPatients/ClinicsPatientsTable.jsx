import {
    Box,
    CircularProgress,
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
  import { ArrowForward, Info, Note } from '@material-ui/icons'
  import MUIDataTable from 'mui-datatables'
  import React, { useState, useEffect } from 'react'
  import { useNavigate } from 'react-router-dom'
  import DeleteIcon from '@material-ui/icons/Delete'
  import { fDateTime } from 'src/utils/formatTime'
  import JobStateStatus from '../common/JobStateStatus'
  import { annotateFile, updateFinishInfo, updateFileNotes, fecthMedicalCenterPatients, deletePatient } from '../../api/file'
  import { PATH_DASHBOARD, ROOTS_PrioVar } from '../../routes/paths'
  import axios from '../../utils/axios'
  import { Link as RouterLink } from 'react-router-dom'
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
  
    const handleClick = () => {
      navigate(`/libra/sample/${fileName}`)
    }
  
    return (
      <Button variant="contained" onClick={handleClick} size="small">
        <ArrowForward/>
      </Button>
    )
  }
  
  const SamplesView = function () {
    //const classes = useStyles()
    let navigate = useNavigate()
    //const filesApi = useFiles()
    //const bedFilesApi = useBedFiles()
    //const { status, data = [] } = filesApi.query
    //const { data: bedFiles = [] } = bedFilesApi.query
    const [data, setData] = useState([])
    const [medicalCenterName, setMedicalCenterName] = useState('');
    const [isLoading, setIsLoading] = useState(true)
    const [isAnnotationModalOpen, setAnnotationModalOpen] = useState(false)
    const [selectedFile, setSelectedFile] = useState(null)
    const [isPatientDeleted, setIsPatientDeleted] = useState(false)
    const fetchData = async () => {
      setIsLoading(true)
      try {
        const data = await fecthMedicalCenterPatients()
        setData(data)
      } catch (error) {
        console.error('Error fetching clinician files:', error)
      } finally {
        setIsLoading(false)
      }
    }

    const addNewNote = (vcfFileId, note) => {
      updateFileNotes(vcfFileId, note).then(() => {
        fetchData();
      })
    }

    const fetchMedicalCenterName = async () => {
        const medicalCenterId = localStorage.getItem('healthCenterId');
        if (medicalCenterId) {
            try {
                const response = await axios.get(`${ROOTS_PrioVar}/medicalCenter/${medicalCenterId}`);
                setMedicalCenterName(response.data.name); // Assuming the response contains an object with a name property
            } catch (error) {
                console.error('Failed to fetch medical center name:', error);
            }
        }
    };

    useEffect(() => {
      fetchMedicalCenterName();
      fetchData();
    }, [])

    useEffect(() => {
      if (isPatientDeleted) {
        fetchData();
      }
    }, [isPatientDeleted]);

    const setFinishedInfo = (row) => {
      const id = row.vcf_id ? row.vcf_id : row.fastq_pair_id
      updateFinishInfo(id).then(() => {
        //filesApi.refresh()
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
          //filesApi.refresh()
          //setAnnotationModalOpen(false)
      })
    }

    const handleSeeSimilarPatients = (row) => {
        localStorage.setItem('patientId', row.patientId);
        localStorage.setItem('patientName', row.patientName);
        navigate(PATH_DASHBOARD.general.similarPatients, { state: { detail: row } });
    }

    const handleDetails = (row) => {
        // TODO
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
            // eslint-disable-next-line eqeqeq
            const isClinicianSame = row?.clinicianId == localStorage.getItem('clinicianId')
            console.log(isClinicianSame)
            console.log(row?.clinicianId)
            console.log(localStorage.getItem('clinicianId'))
            const handleClickConfirm = () => {
              deletePatient(row.patientId).then(() => {
                setIsPatientDeleted(true);
              });
            }
  
            return (
              <>
                  {isClinicianSame ? (
                      <DeleteFileButton onClickConfirm={handleClickConfirm} />
                  ) : (
                      <button disabled={true} style={{ opacity: 0.5 }}>
                          Unauthorized
                      </button>
                  )}
              </>
          );
          
               
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
        name: 'patientName',
        label: 'Patient Name',
        options: {
          filter: true,
          sort: true,
          customBodyRenderLite: (dataIndex) => {
            const row = data[dataIndex]
            if (!row) return null
            return <Chip label={row.patientName} />
          },
        },
      },
      {
        name: 'clinicianName',
        label: 'Clinician Name',
        options: {
          filter: true,
          sort: true,
          customBodyRenderLite: (dataIndex) => {
            const row = data[dataIndex]
            if (!row) return null
            if (!row.file) return null
            return <Chip label={row.file.clinicianName} />
          },
        },
      },
      {
        name: 'clinicianComments',
        label: 'Clinician Comments',
        options: {
          filter: false,
          sort: false,
          customBodyRenderLite(dataIndex) {
            const row = data[dataIndex];
            if (!row.file) return null;
            const clinicianComments = row ? row.file.clinicianComments : null;
      
            return row ? (
              <ExpandOnClick
                expanded={
                  <div>
                    {clinicianComments.map((comment, index) => (
                        <p>{row.file.clinicianName + ": " + comment}</p>
                    ))}
                    <EditableNote
                        onSave={(notes) => addNewNote(row.file.vcfFileId, notes)}
                        details={{ person: row.file.clinicianName }}
                    />
                  </div>
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
        { isLoading ? 
          (<CircularProgress />) : 
          (
            <>
            <Box display="flex" justifyContent="flex-end" mt={2}> 
            <Button 
                variant="contained" 
                color="info" 
                component={RouterLink} to={PATH_DASHBOARD.general.files}
                size="small"
            >
                <ArrowForward /> 
                Upload VCF File 
            </Button>
            </Box>
            <VariantDasboard2
            open={isAnnotationModalOpen}
            handleButtonChange = {handleButtonChange}
            onClose={() => setAnnotationModalOpen()}
            />
            <MUIDataTable
              title={`All patients of ${medicalCenterName || '...'} health center`}
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
        }
      </>
    )
  }
  
  export default SamplesView
  
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