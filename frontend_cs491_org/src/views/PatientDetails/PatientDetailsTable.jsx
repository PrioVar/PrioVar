import {
    Box,
    Button,
    Typography,
    Grid,
    InputLabel,
    FormControl,
    Select,
    MenuItem,
    IconButton,
    CircularProgress,
    CardContent
  } from '@material-ui/core'
  import axios from 'axios';
  import { ArrowBack, CloseOutlined, Add } from '@material-ui/icons'
  import React, { useState, useEffect } from 'react'
  import { useNavigate } from 'react-router-dom'
  import { fetchDiseases, fetchPhenotypeTerms, deletePhenotypeTerm, addPhenotypeTerm } from '../../api/file'
  import { useHpo } from '../../api/vcf'
  import { useParams } from 'react-router-dom'
  import { ROOTS_PrioVar } from '../../routes/paths'
  import { useSnackbar } from 'notistack5'
  import closeFill from '@iconify/icons-eva/close-fill'
  import { Icon } from '@iconify/react'
  import { MIconButton } from '../../components/@material-extend'
  import { Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle } from '@material-ui/core';
  import Tags from 'src/components/Tags'
  import { HPO_OPTIONS } from 'src/constants'

  const PatientDetailsTable = function () {
    //const classes = useStyles()
    const [options, setOptions] = useState([]); // Store dropdown options
    const [phenotypeTermsLoading, setPhenotypeTermsLoading] = useState(true);
    const [selectedOption, setSelectedOption] = useState('')
    const { patientId, fileId } = useParams();
    const { enqueueSnackbar, closeSnackbar } = useSnackbar()
    const [openDialog, setOpenDialog] = useState(false);
    const [hpoList, setHpoList] = useHpo({ fileId })
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

    const sortedOptions = options.slice().sort((a, b) => {
        // Use localeCompare for string comparison to handle special characters and case sensitivity
        return a.diseaseName.localeCompare(b.diseaseName);
    });
    
    const PhenotypeTerm = ({ term }) => {
        const handleDelete = async () => {
            setPhenotypeTermsLoading(true);
            try {
                const response = await deletePhenotypeTerm(patientId, term.id);
                console.log('SUCCESS: ', response);
            }
            catch (error) {
                console.error('FAILURE');
                console.error('Error deleting term:', error);
            }
            try {
                const data = await fetchPhenotypeTerms(patientId);
                setDetails({ ...details, phenotypeTerms: data });
                console.log("patientId: ", patientId)
                console.log(data);
                console.log('SUCCESS')
            }
            catch (error) {
                console.error('FAILURE');
                console.error('Error refetching terms:', error);
            }
            setPhenotypeTermsLoading(false);
        };
      
        return (
          <div style={{ border: '1px solid #ccc', borderRadius: '5px', padding: '5px', display: 'inline-flex', alignItems: 'center', marginRight: '5px' }}>
            <Typography variant="body1" style={{ marginRight: '5px' }}>{term.name}</Typography>
            <IconButton size="small" onClick={handleDelete}>
              <CloseOutlined color='error' />
            </IconButton>
          </div>
        );
    };

    // Fetch dropdown options
    useEffect(() => {
        fecthData()
    }, []);

    // Handle option selection
    const handleSelectionChange = (event) => {
        setSelectedOption(event.target.value);
    };

    const handleSubmit = () => {
        if (details.disease?.diseaseName && details.disease.diseaseName !== '') {
            // If disease is already set, ask for confirmation
            setOpenDialog(true);
        } else {
            // If no disease is set, proceed to change it
            changeDisease();
        }
    };
    // KAAN LOOK TO THIS FUNCTION
    const addPhenotype = async () => {
        console.log('Phenotype Add');
        const phenotypeTerms = hpoList.map(item => {
            // Extract the numeric part of the HP code and remove leading zeros
            const id = parseInt(item.value.split(':')[1], 10);
            return { id };
          });
          const phenotypeTermIds = phenotypeTerms.map(term => term.id);
          const formData = new FormData();
        formData.append('phenotypeTermIds', phenotypeTermIds);
        if( phenotypeTerms.length > 0){
            setPhenotypeTermsLoading(true);
            try {
                const response = await addPhenotypeTerm(patientId, formData); // IMPLEMENT THE BACKEND SERVICE
                console.log("SUCCESS: ", response)
            } catch (error) {
                console.log("FAILURE")
                console.error('Error posting data:', error);
                // Handle error notification
            }
            try {
                const data = await fetchPhenotypeTerms(patientId);
                setDetails({ ...details, phenotypeTerms: data });
                console.log('SUCCESS')
            }
            catch (error) {
                console.error('FAILURE');
                console.error('Error refetching terms:', error);
            }
            setPhenotypeTermsLoading(false);
        }
    };

    const ManageHpo = function ({ hpoList, setHpoList}) {
  
        return <Tags title="Symptoms" options={HPO_OPTIONS} value={hpoList} onChange={setHpoList} />
      }

    const changeDisease = async () => {
        try {
            console.log(selectedOption)
            const response = await axios.post(`${ROOTS_PrioVar}/patient/${patientId}/addDisease/${selectedOption}`);
            console.log(response.data);

            // Fetch updated patient details
            await fetchPatientDetails();
            enqueueSnackbar('Disease Added Successfully!', {
                variant: 'success',
                action: (key) => (
                  <MIconButton size="small" onClick={() => closeSnackbar(key)}>
                    <Icon icon={closeFill} />
                  </MIconButton>
                ),
            })
        } catch (error) {
            console.error('Error posting data:', error);
            // Handle error notification
        }
    };

    const fetchPatientDetails = async () => {
        try {
            const response = await axios.get(`${ROOTS_PrioVar}/patient/${patientId}`);
            console.log("SUCCESS", response.data);
            setDetails(response.data);
            setPhenotypeTermsLoading(false);
        } catch (error) {
            console.error('Error fetching patient details:', error);
        }
    };

    useEffect(() => {
        fetchPatientDetails();
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
                    { phenotypeTermsLoading ? (<CircularProgress />)
                    :
                    ( <>
                        {details.phenotypeTerms.map((term, index) => (
                        <PhenotypeTerm key={index} term={term} />
                        ))}
                        </>)}
                </Grid>
                <Grid item xs={4} mt={4}>
                    <Typography variant="h6">Add Phenotype Terms</Typography>
                    <CardContent>
                        <ManageHpo hpoList={hpoList} setHpoList={setHpoList}/>
                    </CardContent>
                    <Button onClick={() => addPhenotype()} color="primary" variant="contained" sx={{ mt: 0.5 }}>
                        Add Phenotype Term
                    </Button>
                </Grid>
                <Grid item xs={4} >
                    <Typography variant="h6">Set Diagnosis</Typography>
                    <FormControl fullWidth variant="outlined">
                    <InputLabel>Select Disease</InputLabel>
                    <Select
                        value={selectedOption}
                        onChange={(e) => setSelectedOption(e.target.value)}
                        label="Select Disease"
                    >
                        {sortedOptions.map((option) => (
                        <MenuItem key={option.id} value={option.id}>
                            {option.diseaseName}
                        </MenuItem>
                        ))}

                    </Select>
                    </FormControl>
                    <Button onClick={() => handleSubmit(selectedOption)} color="primary" variant="contained" sx={{ mt: 0.5 }}>
                    Set Diagnosis
                    </Button>
                </Grid>
            </Grid>
        </Box>
        <Dialog
            open={openDialog}
            onClose={() => setOpenDialog(false)}
        >
            <DialogTitle>{"Change Diagnosis"}</DialogTitle>
            <DialogContent>
                <DialogContentText>
                    Are you sure you want to change the diagnosis?
                </DialogContentText>
            </DialogContent>
            <DialogActions>
                <Button onClick={() => setOpenDialog(false)} color="primary">
                    Cancel
                </Button>
                <Button onClick={() => {
                    setOpenDialog(false);
                    changeDisease();
                }} color="primary" autoFocus>
                    Yes
                </Button>
            </DialogActions>
        </Dialog>
        </>
    )

  }
  
  export default PatientDetailsTable
  