import {
  Box,
  CircularProgress,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  IconButton,
  DialogContentText,
  Typography,
  TextField,
} from '@material-ui/core'
import { makeStyles } from '@material-ui/styles'
import { ArrowForward, Info, Note } from '@material-ui/icons'
import Tooltip from '@mui/material/Tooltip';
import MUIDataTable from 'mui-datatables'
import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import DeleteIcon from '@material-ui/icons/Delete'
import { fDateTime } from 'src/utils/formatTime'
import AssessmentIcon from '@mui/icons-material/Assessment';
//import { useFiles, annotateFile, useBedFiles, updateFinishInfo, updateFileNotes } from '../../api/file'
import { fecthClinicianFiles, fecthMedicalCenterFiles, annotateFile, updateFileNotes, deleteVCF } from '../../api/file'

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
    navigate(`/priovar/sample/${fileId}/${sampleName}`)
  }

  return (
    <Button variant="contained" onClick={handleClick} size="small">
      <ArrowForward />
    </Button>
  )
}

const SamplesView = function ({ isFileUploaded, resetFileUploaded }) {
  const [data, setData] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [isAnnotationModalOpen, setAnnotationModalOpen] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const [isFileDeleted, setIsFileDeleted] = useState(false)
  const navigate = useNavigate()

  const fetchData = async () => {
    setIsLoading(true)
    try {
      var data = null
      // eslint-disable-next-line eqeqeq
      if (localStorage.getItem('clinicianId') == -1) { 
        console.log("muhahaha")
        data = await fecthMedicalCenterFiles()
      }
      else {
        data = await fecthClinicianFiles()
      }
      console.log(data)
      setData(data)
    } catch (error) {
      console.error('Error fetching clinician files:', error)
    } finally {
      setIsLoading(false)
      resetFileUploaded();  // Call to reset the upload state
    }
  }

  const addNewNote = (vcfFileId, note) => {
    updateFileNotes(vcfFileId, note).then(() => {
      fetchData();
    })
  }

  // Effect for initial data load
  useEffect(() => {
      fetchData();
  }, []); // Empty dependency array to run only once on mount

  useEffect(() => {
    if (isFileUploaded) {
        fetchData();
    }
  }, [isFileUploaded]);

  useEffect(() => {
    if (isFileDeleted) {
        fetchData();
    }
  }, [isFileDeleted]);

  const navigateToVariantDashboard = (fileName) => {
    navigate(`/priovar/sample/${fileName}`)
  }

  const handleAnnotationModelOpen = (row) => {
    setSelectedFile(row)
    setAnnotationModalOpen(true)
  }
  const handleAnnotationModalClose = (newOpenValue) => {
    setAnnotationModalOpen(newOpenValue);
  };
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
        setAnnotationModalOpen(false)
      })
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
            deleteVCF(row.vcfFileId).then(() => {
              setIsFileDeleted(true);
            });
          }

          return <DeleteFileButton onClickConfirm={handleClickConfirm} />
        },
      },
    },
    {
      name: 'clinicianName',
      label: 'Clinician Name',
      options: {
        filter: true,
        sort: true,
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
          return row ? fDateTime(row.createdAt) : null;

        },
      },
    },
    {
      name: 'completed_at',
      label: 'Completed At',
      options: {
        filter: false,
        sort: true,
        customBodyRenderLite(dataIndex) {
          const row = data[dataIndex]
          return row.finishedAt? fDateTime(row.finishedAt) : <>Not Finished</>;

        },
      },
    },
    {
      name: 'fileName',
      label: 'File Name',
      options: {
        filter: true,
        sort: true,
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
          const clinicianComments = row ? row.clinicianComments : null;
    
          return row ? (
            <ExpandOnClick
              expanded={
                <div>
                  {clinicianComments.map((comment, index) => (
                      <p>{row.clinicianName + ": " + comment}</p>
                  ))}
                  <EditableNote
                      onSave={(notes) => addNewNote(row.vcfFileId, notes)}
                      details={{ person: row.clinicianName }}
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
      name: 'annotate',
      label: 'Annotate VCF File',
      options: {
        filter: false,
        sort: false,
        customBodyRenderLite: (dataIndex) => {
          const row = data[dataIndex]
          //const status = getStatusLabel(row)
          const status = row?.fileStatus
          const isAnnotated = status === 'FILE_ANNOTATED'
          if (status === 'ANALYSIS_DONE') {
            return (
              <Tooltip title="Analysis complete. Click to view results.">
                <span> {/* Tooltip children need to be able to hold a ref */}
                  <Button size="small" variant="contained" color="primary" onClick={() => navigateToVariantDashboard(row.fileName)}>
                    <AssessmentIcon /> View Results
                  </Button>
                </span>
              </Tooltip>
            );
          }
          if (status === 'ANALYSIS_IN_PROGRESS') {
            return (
                <Tooltip title="Analysis in progress">
                    <div style={{ display: 'flex', alignItems: 'center' }}>
                    <CircularProgress size={24} />
                    <Typography variant="body2" style={{ marginLeft: 8 }}>Analysis in progress</Typography>
                    </div>
                </Tooltip>
            );
          }
          return (
            <Tooltip title={isAnnotated ? "File already annotated" : "Click to annotate file"}>
                <span> {/* Tooltip children need to be able to hold a ref */}
                    <Button disabled={isAnnotated} onClick={() => handleAnnotationModelOpen(row)} size="small" variant="contained" color="info">
                        <Info />  Annotate File
                    </Button>
                </span>
            </Tooltip>
         )
        },
      },
    },
    // Define additional columns as needed based on the keys in the data objects
  ]
/*
  if (isLoading) {
    return <CircularProgress />
  }
*/
  return (
    <>
      {isLoading ? (
        <CircularProgress />
      ) : (
        <>
          <VariantDasboard2
            open={isAnnotationModalOpen}
            handleButtonChange={handleButtonChange}
            onClose={() => setAnnotationModalOpen()}
            vcfFileId={selectedFile?.vcfFileId}
            fetchData={fetchData}
            setAnnotationModalOpen={setAnnotationModalOpen}
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
      )}
    </>
  );

  
}

export default SamplesView