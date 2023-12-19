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
    Grid,
    TextField,
    Tab,
    Tabs,
    Tooltip,
    InputLabel,
    FormControl,
    Input,
    Select,
    MenuItem
  } from '@material-ui/core'
  import axios from 'axios';
  import { makeStyles } from '@material-ui/styles'
  import { ArrowForward, Info, Note, Add } from '@material-ui/icons'
  import MUIDataTable from 'mui-datatables'
  import React, { useState, useMemo, useEffect } from 'react'
  import { useNavigate } from 'react-router-dom'
  import DeleteIcon from '@material-ui/icons/Delete'
  import { fDateTime } from 'src/utils/formatTime'
  import JobStateStatus from '../common/JobStateStatus'
  import { deleteVcfFile } from '../../api/vcf'
  import { deleteFastqFile } from '../../api/fastq'
  import { useFiles, annotateFile, useBedFiles, updateFinishInfo, updateFileNotes } from '../../api/file'
  import { PATH_DASHBOARD } from '../../routes/paths'
  import { Link as RouterLink } from 'react-router-dom'
  import ExpandOnClick from 'src/components/ExpandOnClick'
  import AnalysedCheckbox from '../common/AnalysedCheckbox'
  import { useParams } from 'react-router-dom'

  import Tags from 'src/components/Tags'
  // api utils
  import { updateTrio, useHpo } from '../../api/vcf'
  // constants
  import { HPO_OPTIONS, DASHBOARD_CONFIG } from 'src/constants'
  
  const PatientDetailsTable = function () {
    //const classes = useStyles()
    const bedFilesApi = useBedFiles()
    const { data: bedFiles = [] } = bedFilesApi.query
    const [isAnnotationModalOpen, setAnnotationModalOpen] = useState(false)
    const [selectedFile, setSelectedFile] = useState(null)
    const { fileId, sampleName } = useParams()
    const filesApi = useFiles()
    const { status, data = [] } = filesApi.query
    const fileDetails = useMemo(
      () => data.find((f) => f.vcf_id === fileId || f.fastq_pair_id === fileId),
      [data, fileId, filesApi],
    )

    //
    const [details, setDetails] = useState(
        { name: '', age: '', sex: '', disease: '', assignedClinic: '', phenotypeTerms: [] }
    );

    useEffect(() => {
        // Ali Veli patient id: 17700
        axios.get(`http://localhost:8080/patient/17700`)
          .then(response => {
            console.log("SUCCESS")
            console.log(response.data)
            setDetails(response.data);
          })
          .catch(error => console.error('Error fetching data:', error));
      }, []);

    return (
        <>

    <Box p={3}>
        <Typography variant="h4" align="center">Patient Details</Typography>
        <Grid container spacing={2} mt={4}>
            {/* Name, Age, Sex, and Assigned Clinic in one row */}
            <Grid item xs={3}>
            <Typography variant="h6">Name: </Typography> {details.name}
            </Grid>
            <Grid item xs={3}>
            <Typography variant="h6">Age: </Typography>{details.age}
            </Grid>
            <Grid item xs={3}>
            <Typography variant="h6">Sex: </Typography>{details.sex}
            </Grid>
            <Grid item xs={3}>
            <Typography variant="h6">Health Center: </Typography>{details.medicalCenter?.name}
            </Grid>

            {/* Disease and Phenotype Terms in the next row */}
            <Grid item xs={6} mt={4}>
            <Typography variant="h6">Disease: </Typography> {details.disease.diseaseName}
            </Grid>
            <Grid item xs={6} mt={4}>
            <Typography variant="h6">Phenotype Terms:</Typography>
                {details.phenotypeTerms.map((term, index) => (
                    <div key={index}>{term.name}</div>
          ))}
            </Grid>
        </Grid>
    </Box>
        </>
    )

  }
  
  export default PatientDetailsTable
  