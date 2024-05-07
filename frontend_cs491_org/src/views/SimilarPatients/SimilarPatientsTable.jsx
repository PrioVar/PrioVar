import {
    Box, CircularProgress, Button, Typography, Grid, InputLabel, FormControl, Input, Snackbar, Tabs, Tab,
    Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle, TextField, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper
  } from '@material-ui/core';
  import { Info, ArrowBack } from '@material-ui/icons';
  import axios from 'axios';
  import React, { useState, useEffect } from 'react';
  import { useFiles, useBedFiles, sendInformationRequest, fetchAllAvailablePatients, fetchWaitingInformationRequests } from '../../api/file';
  import { ROOTS_PrioVar } from '../../routes/paths';
  import { useParams, useNavigate } from 'react-router-dom';
  import { Link as RouterLink } from 'react-router-dom';
  import { PATH_DASHBOARD } from '../../routes/paths';
  import { useSnackbar } from 'notistack5'
  import closeFill from '@iconify/icons-eva/close-fill'
  import { Icon } from '@iconify/react';
  import { MIconButton } from '../../components/@material-extend';
  

  const SimilarPatientsTable = function () {
    const navigate = useNavigate();
    const patientId = localStorage.getItem('patientId');
    const patientName = localStorage.getItem('patientName');
    const clinicianId = localStorage.getItem('clinicianId'); 
    const [currentPatientId, setCurrentPatientId] = useState('');
    const [requestDescription, setRequestDescription] = useState('');
    const [openDialog, setOpenDialog] = useState(false);
    const [allAvailablePatients, setAllAvailablePatients] = useState([]);
    const [waitingRequests, setWaitingRequests] = useState([]);
    const { enqueueSnackbar, closeSnackbar } = useSnackbar();

    const bedFilesApi = useBedFiles();
    const { data: bedFiles = [] } = bedFilesApi.query;
    const { fileId, sampleName } = useParams();
    const filesApi = useFiles();
    const { status, data = [] } = filesApi.query;
    const fileDetails = data.find((f) => f.vcf_id === fileId || f.fastq_pair_id === fileId);
  
    const [resultCount, setResultCount] = useState(fileDetails?.details?.resultCount);
    const [rows, setRows] = useState([]);
    const [searching, setSearching] = useState(false);
    const [reports, setReports] = useState([]);
    const [selectedTab, setSelectedTab] = useState(0);
  
    const handleResultCount = (e) => setResultCount(e.target.value);
    const handleTabChange = (event, newValue) => setSelectedTab(newValue);
  
    const handleRequestOpen = (secondaryPatientId) => {
        setCurrentPatientId(secondaryPatientId); // Set the current patient ID when opening the dialog
        setOpenDialog(true);
    };
    const handleRequestClose = () => setOpenDialog(false);

const handleRequestSubmit = async () => {
    try {
        handleRequestClose();
        enqueueSnackbar('Request sent successfully!', {
            variant: 'success',
            action: (key) => (
                <MIconButton size="small" onClick={() => closeSnackbar(key)}>
                    <Icon icon={closeFill} />
                </MIconButton>
            ),
        });
        await sendInformationRequest(clinicianId, currentPatientId, requestDescription);
     
        // Close the dialog
        // Update the waiting requests array
        const newRequest = { patient: { id: currentPatientId } }; // Structure this as per your actual data model
        setWaitingRequests([...waitingRequests, newRequest]);
    } catch (error) {
        console.error('Error sending request:', error.response);
        enqueueSnackbar("Failed to send request", {
            variant: 'error',
            action: (key) => (
                <MIconButton size="small" onClick={() => closeSnackbar(key)}>
                    <Icon icon={closeFill} />
                </MIconButton>
            ),
        });
    }
};
  
    const fetchReports = async () => {
        try {
            const response = await axios.get(`${ROOTS_PrioVar}/similarityReport/byPatient/${patientId}/3`);
            setReports(response.data);
        } catch (error) {
            console.error('Error fetching reports:', error.response);
        }
    };
  
    const fetchAllPatients = async () => {
        const allPatients = await fetchAllAvailablePatients();
        setAllAvailablePatients(allPatients);

    };

    const fetchWaitingRequests = async () => {
        const waitingRequests = await fetchWaitingInformationRequests(clinicianId);
        setWaitingRequests(waitingRequests);
    };

  
    useEffect(() => {
        fetchReports();
        fetchAllPatients(); // Fetch all available patients when the component mounts
        fetchWaitingRequests();
    }, []);


    const isPatientAvailable = (secondaryPatientId) => {
        return allAvailablePatients.some(patient => patient.patientId === secondaryPatientId);
    };

    const isPatientRequested = (secondaryPatientId) => {
        return waitingRequests.some(request => request.patient.id === secondaryPatientId);
    };

  
    const handleSearch = async () => {
        try {
            setSearching(true);
            const response = await axios.get(`${ROOTS_PrioVar}/similarityReport/mostSimilarPatients/${patientId}/${resultCount}`);
            setRows(response.data.pairSimilarities);
            fetchReports(); // Update reports after generating a new report
        } catch (error) {
            console.error('Similar patients query error:', error.response);
        } finally {
            setSearching(false);
        }
    };
  
    return (
        <>
            <Button onClick={() => navigate(-1)} sx={{ ml: 1, mt: 3 }}>
                <ArrowBack sx={{ mr: 1 }} /> Go Back To Patients
            </Button>
            <Box p={3} mt={4}>
                <Typography variant="h5">Similar Patients for {patientName}</Typography>
                <Grid container spacing={2} alignItems="flex-end" mt={4}>
                    <Grid item xs={3}>
                        <FormControl fullWidth sx={{ width: '100%' }}>
                            <InputLabel style={{ color: 'black' }} id="select-age">Number of similar patients</InputLabel>
                            <Input type="number" value={resultCount} onChange={handleResultCount}></Input>
                        </FormControl>
                    </Grid>
                    <Grid item container xs={12} sm={6} spacing={2}>
                        <Grid item xs={6}>
                            <Button variant="contained" color="primary" onClick={handleSearch}>
                                Search
                            </Button>
                        </Grid>
                    </Grid>
                </Grid>
                <Box mt={2}>
                    <Tabs value={selectedTab} onChange={handleTabChange} variant="fullWidth">
                        {reports.map((report, index) => (
                            <Tab label={`Report ${index + 1}`} key={index} />
                        ))}
                    </Tabs>
                    {reports.length > 0 ? (
                        <TableContainer component={Paper}>
                            <Table>
                            <TableHead>
                              <TableRow>
                                  <TableCell>Name</TableCell>
                                  <TableCell align="right">Age</TableCell>
                                  <TableCell align="right">Sex</TableCell>
                                  <TableCell align="right">Similarity Score</TableCell>
                                  <TableCell align="right">Patient Details</TableCell> {/* Empty on purpose */}
                              </TableRow>
                              </TableHead>
                              <TableBody>
                              {reports[selectedTab]?.pairSimilarities.map((row, index) => (
                                <TableRow key={index}>
                                    <TableCell component="th" scope="row">
                                        {isPatientAvailable(row.secondaryPatient.id) ? row.secondaryPatient.name : "************"}
                                    </TableCell>
                                    <TableCell align="right">{row.secondaryPatient.age}</TableCell>
                                    <TableCell align="right">{row.secondaryPatient.sex}</TableCell>
                                    <TableCell align="right">{row.phenotypeScore.toFixed(2)}</TableCell>
                                    <TableCell align="right">
                                        
                                        {!isPatientAvailable(row.secondaryPatient.id) &&  !isPatientRequested(row.secondaryPatient.id) && (
                                            <Button 
                                                variant="contained" 
                                                color="info" 
                                                size="small" 
                                                onClick={() => handleRequestOpen(row.secondaryPatient.id)}
                                            >
                                                <Info sx={{ marginRight: '8px' }}/> Request
                                            </Button>
                                        )}
                                        {isPatientAvailable(row.secondaryPatient.id) && (
                                        <Button 
                                            variant="contained" 
                                            color="success" // Green button to indicate availability
                                            size="small" 
                                            onClick={() => navigate(PATH_DASHBOARD.general.patientDetailsConst.replace(':patientId', row.secondaryPatient.id))} // Using placeholders for dynamic URL
                                        >
                                            <Info sx={{ marginRight: '8px' }}/> View
                                        </Button>
                                    )}
                                       {isPatientRequested(row.secondaryPatient.id) &&  !isPatientAvailable(row.secondaryPatient.id) && (
                                            <Button
                                                variant="contained"
                                                color="warning"
                                                size="small"
                                                disabled
                                            >
                                                <Info sx={{ marginRight: '8px' }} /> Already Requested
                                            </Button>
                                    )}
                                    </TableCell>
                                </TableRow>
                            ))}
                              </TableBody>
                            </Table>
                        </TableContainer>
                    ) : (
                        <Typography variant="subtitle1" style={{ textAlign: 'center', marginTop: '20px' }}>
                            No record found
                        </Typography>
                    )}
                </Box>
  
                {/* Snackbar for "Searching..." message */}
                <Snackbar
                    anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
                    open={searching}
                    message="Searching..."
                    action={<CircularProgress color="inherit" />}
                />
            </Box>
            <Dialog open={openDialog} onClose={handleRequestClose}>
                <DialogTitle>Send Information Request</DialogTitle>
                <DialogContent>
                    <DialogContentText>
                        Please enter your justification for requesting the information of this patient:
                    </DialogContentText>
                    <TextField
                        autoFocus
                        margin="dense"
                        id="name"
                        label="Request Justification"
                        type="text"
                        fullWidth
                        variant="outlined"
                        value={requestDescription}
                        onChange={(e) => setRequestDescription(e.target.value)}
                    />
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleRequestClose} color="primary">
                        Cancel
                    </Button>
                    <Button onClick={handleRequestSubmit} color="primary">
                        Submit
                    </Button>
                </DialogActions>
            </Dialog>
        </>
    );
  };
  
  export default SimilarPatientsTable;