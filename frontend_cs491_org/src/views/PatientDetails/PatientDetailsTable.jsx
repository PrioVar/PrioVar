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
    const [options, setOptions] = useState([]); // Store dropdown options
    const [selectedOption, setSelectedOption] = useState('')
    const [patientId, setPatientId] = useState('')
    let navigate = useNavigate();

    //
    const [details, setDetails] = useState(
        { name: '', age: '', sex: '', disease: '', assignedClinic: '', phenotypeTerms: [] }
    );

    // Fetch dropdown options
    useEffect(() => {
        const fetchOptions = async () => {
            try {
                const response = await axios.get(`http://localhost:8080/disease`);
                const fetchedOptions = response.data;
                setOptions(fetchedOptions);
            } catch (error) {
                console.error('Error fetching options:', error);
            }
        };
        fetchOptions();
    }, []);

    // Handle option selection
    const handleSelectionChange = (event) => {
        setSelectedOption(event.target.value);
    };

      // Handle submit
    const handleSubmit = async () => {
        try {
            console.log(selectedOption)
            const patient = await getPatient()
            const response = await axios.post(`http://localhost:8080/patient/${patient.data.id}/addDisease/${selectedOption}`);
            console.log(response.data);
            // Handle response or success notification
            navigate(0)
        } catch (error) {
            console.error('Error posting data:', error);
            // Handle error notification
        }
    };

    const getPatient = async () => {

        try {
            return axios.get(`http://localhost:8080/patient/getPatient`);
        } catch (error) {
            console.log(error.response)
        }
    }

    useEffect(() => {
        const fetch = async () => {
        // first get the patient
        const patient = await getPatient()
        setPatientId(patient.data.id)

        // Ali Veli patient id: 17700
        axios.get(`http://localhost:8080/patient/${patient.data.id}`)
          .then(response => {
            console.log("SUCCESS")
            console.log(response.data)
            setDetails(response.data);
          })
          .catch(error => console.error('Error fetching data:', error));
        }
        fetch();

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
            <Grid item xs={4} mt={4}>
            <Typography variant="h6">Disease: </Typography> {details.disease.diseaseName}
            </Grid>
            <Grid item xs={4} mt={4}>
            <Typography variant="h6">Phenotype Terms:</Typography>
                {details.phenotypeTerms.map((term, index) => (
                    <div key={index}>{term.name}</div>
          ))}
            </Grid>
            <Grid item xs={4} mt={4}>
            <FormControl fullWidth variant="outlined">
                <InputLabel>Select Disease</InputLabel>
                <Select
                    value={selectedOption}
                    onChange={(e) => setSelectedOption(e.target.value)}
                    label="Select Disease"
                >
                    {options.map((option) => (
                    <MenuItem key={option.id} value={option.id}>
                        {option.diseaseName}
                    </MenuItem>
                    ))}
                </Select>
            </FormControl>
            <Button onClick={() => handleSubmit(selectedOption)} color="primary" variant="contained">
            Set Diagnosis
            </Button>
            </Grid>
        </Grid>
    </Box>
        </>
    )

  }
  
  export default PatientDetailsTable
  