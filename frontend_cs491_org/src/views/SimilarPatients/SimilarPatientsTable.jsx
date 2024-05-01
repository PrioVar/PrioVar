import {
    Box,
    CircularProgress,
    Button,
    Typography,
    Grid,
    InputLabel,
    FormControl,
    Input,
    Snackbar,
    Tabs,
    Tab,
  } from '@material-ui/core';
  import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper } from '@material-ui/core';
  import axios from 'axios';
  import { Info, ArrowBack } from '@material-ui/icons';
  import React, { useState, useEffect } from 'react';
  import { useFiles, useBedFiles } from '../../api/file';
  import { ROOTS_PrioVar } from '../../routes/paths';
  import { useParams, useNavigate } from 'react-router-dom';
  
  const SimilarPatientsTable = function () {
    const patientId = localStorage.getItem('patientId');
    const patientName = localStorage.getItem('patientName');
    const bedFilesApi = useBedFiles();
    const { data: bedFiles = [] } = bedFilesApi.query;
    const { fileId, sampleName } = useParams();
    const filesApi = useFiles();
    let navigate = useNavigate();
    const { status, data = [] } = filesApi.query;
    const fileDetails = data.find((f) => f.vcf_id === fileId || f.fastq_pair_id === fileId);
  
    const [resultCount, setResultCount] = useState(fileDetails?.details?.resultCount);
    const [rows, setRows] = useState([]);
    const [searching, setSearching] = useState(false);
    const [reports, setReports] = useState([]);
    const [selectedTab, setSelectedTab] = useState(0);
  
    const handleResultCount = (e) => setResultCount(e.target.value);
  
    const handleTabChange = (event, newValue) => setSelectedTab(newValue);
  
    const fetchReports = async () => {
      try {
        const response = await axios.get(`${ROOTS_PrioVar}/similarityReport/byPatient/${patientId}/3`);
        setReports(response.data);
      } catch (error) {
        console.error('Error fetching reports:', error.response);
      }
    };
  
    useEffect(() => {
      fetchReports();
    }, []);
  
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
          <Typography variant="h5">Similar Patients for {patientName} </Typography>
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
                      <TableCell align="right">Request</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {reports[selectedTab]?.pairSimilarities.map((row, index) => (
                      <TableRow key={index}>
                        <TableCell component="th" scope="row">
                          ************
                        </TableCell>
                        <TableCell align="right">{row.secondaryPatient.age}</TableCell>
                        <TableCell align="right">{row.secondaryPatient.sex}</TableCell>
                        <TableCell align="right">{row.phenotypeScore.toFixed(2)}</TableCell>
                        <TableCell align="right">
                          <Button variant="contained" color="info" size="small">
                            <Info />
                          </Button>
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
      </>
    );
  };
  
  export default SimilarPatientsTable;
  