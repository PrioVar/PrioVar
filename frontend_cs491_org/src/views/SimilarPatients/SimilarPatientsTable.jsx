import {
    Box,
    CircularProgress,
    Button,
    Typography,
    Grid,
    InputLabel,
    FormControl,
    Input,
    Snackbar
  } from '@material-ui/core'
  import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper } from '@material-ui/core';
  import axios from 'axios';
  import { Info, ArrowBack } from '@material-ui/icons'
  import React, { useState, useMemo } from 'react'
  import { useFiles, useBedFiles } from '../../api/file'
  import { ROOTS_PrioVar } from '../../routes/paths'
  import { useParams, useNavigate } from 'react-router-dom'
  
  const SimilarPatientsTable = function () {

    const patientId = localStorage.getItem('patientId')
    const patientName = localStorage.getItem('patientName')
    //const classes = useStyles()
    const bedFilesApi = useBedFiles()
    const { data: bedFiles = [] } = bedFilesApi.query
    const { fileId, sampleName } = useParams()
    const filesApi = useFiles()
    let navigate = useNavigate();
    const { status, data = [] } = filesApi.query
    const fileDetails = useMemo(
      () => data.find((f) => f.vcf_id === fileId || f.fastq_pair_id === fileId),
      [data, fileId, filesApi],
    )

    //
    const [resultCount, setResultCount] = useState(fileDetails?.details?.resultCount)
    const [rows, setRows] = useState([]);
    const [searching, setSearching] = useState(false);
    const handleResultCount = (e) => setResultCount(e.target.value)

    const handleSearch = async () => {

        try {
            setSearching(true);

            const response = await axios.get(`${ROOTS_PrioVar}/similarityReport/mostSimilarPatients/${patientId}/${resultCount}`);
            console.log("SUCCESS!")
            //console.log(response)
            setRows(response.data.pairSimilarities);
    
          } catch (error) {
            console.log("FAIL!")
            console.error('similar patients query error:', error.response);
          }
          finally {
            setSearching(false);
          }
      };

    return (
        <>
        <Button onClick={() => navigate(-1)} sx={{ ml:1, mt: 3 }}>
            <ArrowBack sx={{ mr: 1 }} /> Go Back To Patients
        </Button>
    <Box p={3} mt={4}>
    <Typography variant="h5">Similar Patients for {patientName} </Typography>
      <Grid container spacing={2} alignItems="flex-end" mt={4}>
        <Grid item xs={2}>
            <FormControl fullWidth>
                <InputLabel style={{ color: 'black' }} id="select-age">No of similar patients</InputLabel>
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
        <Box mt={12}>
      </Box>
    </Box>

        {/* Snackbar for "Searching..." message */}
    <Snackbar
    anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
    open={searching}
    message="Searching..."
    action={<CircularProgress color="inherit" />}
    />

    <Box p={3} mt={4}>
    
    {rows.length > 0 ? (
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
                {rows.map((row) => (
                    <TableRow key={row.secondaryPatient.name + row.phenotypeScore}>
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




        

        </>
    )

  }
  
  export default SimilarPatientsTable
  