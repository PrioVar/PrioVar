import {
    Box,
    Button,
    Typography,
    Grid,
    InputLabel,
    FormControl,
    Select,
    MenuItem
  } from '@material-ui/core'
  import axios from 'axios';
  import { ArrowBack, } from '@material-ui/icons'
  import React, { useState, useMemo, useEffect } from 'react'
  import { useNavigate } from 'react-router-dom'
  import { useFiles, useBedFiles, fetchDiseases } from '../../api/file'
  import { Link as RouterLink } from 'react-router-dom'
  import { useParams } from 'react-router-dom'
  
  const PatientDetailsTable = function () {
    //const classes = useStyles()
    const bedFilesApi = useBedFiles()
    const { data: bedFiles = [] } = bedFilesApi.query
    const { fileId, sampleName } = useParams()
    const filesApi = useFiles()
    const { status, data = [] } = filesApi.query
    const fileDetails = useMemo(
      () => data.find((f) => f.vcf_id === fileId || f.fastq_pair_id === fileId),
      [data, fileId, filesApi],
    )
    const [options, setOptions] = useState([]); // Store dropdown options
    const [selectedOption, setSelectedOption] = useState('')
    const { patientId } = useParams();
    let navigate = useNavigate();

    //
    const [details, setDetails] = useState(
        { name: '', age: '', sex: '', disease: '', assignedClinic: '', phenotypeTerms: [] }
    );

    const fecthData = async () => {
        try {
            const fetchedDiseases = await fetchDiseases();
            setOptions(fetchedDiseases);
        } catch (error) {
            console.error('Error fetching options:', error);
        }
    };
    // Fetch dropdown options
    useEffect(() => {
        fecthData()
    }, []);

    // Handle option selection
    const handleSelectionChange = (event) => {
        setSelectedOption(event.target.value);
    };

      // Handle submit
    const handleSubmit = async () => {
        try {
            console.log(selectedOption)
            const response = await axios.post(`http://localhost:8080/patient/${patientId}/addDisease/${selectedOption}`);
            console.log(response.data);
            // Handle response or success notification
            navigate(0)
        } catch (error) {
            console.error('Error posting data:', error);
            // Handle error notification
        }
    };

    useEffect(() => {
        const fetch = async () => {
            axios.get(`http://localhost:8080/patient/${patientId}`)
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
        <Button onClick={() => navigate(-1)} sx={{ ml:1, mt: 3 }}>
            <ArrowBack sx={{ mr: 1 }} /> Go Back To Patients
        </Button>
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
            <Typography variant="h6">Disease: </Typography> {details.disease?.diseaseName}
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
  