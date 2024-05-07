import {
  Box,
  Button,
  Typography,
  Grid,
  TextField,
  InputLabel,
  FormControl,
  Select,
  MenuItem,
} from '@material-ui/core'
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper } from '@material-ui/core';
import axios from 'axios';
import React, { useState } from 'react'
import { useParams } from 'react-router-dom'
import Page from 'src/components/Page'

import Tags from 'src/components/Tags'
// api utils
import { useHpo } from '../../api/vcf'
// constants
import { HPO_OPTIONS } from 'src/constants'

const CustomQueryTable = function () {
  //const classes = useStyles()
  const { fileId, sampleName } = useParams()

  //
  const [gene, setGene] = useState([]);
  const [ageIntervalStart, setAgeIntervalStart] = useState('');
  const [ageIntervalEnd, setAgeIntervalEnd] = useState('');
  const [gender, setGender] = useState(''); 
  const [rows, setRows] = useState([]);

  const geneOptions = ['ABCA1', 'ABCA2', 'ABCA3', 'ABCA4', 'ABCB7', 'ABAT', 'ABL1', 'NAT2', 'AARS1'];

  // Sort the geneOptions alphabetically
  const sortedGeneOptions = geneOptions.slice().sort((a, b) => {
    // Use localeCompare for string comparison to handle special characters and case sensitivity
    return a.localeCompare(b);
  });

  const handleSearch = async () => {

      const phenotypeTerms = hpoList.map(item => {
          // Extract the numeric part of the HP code and remove leading zeros
          const id = parseInt(item.value.split(':')[1], 10);
          return { id };
      });

      const requestBody = {
          sex: gender,
          genes: gene.map(g => ({ geneSymbol: g })),
          phenotypeTerms: phenotypeTerms,
          ageIntervalStart: ageIntervalStart,
          ageIntervalEnd: ageIntervalEnd
      };

      //console.log("body:")
      //console.log(requestBody)
      try {
          console.log('Sending custom query...')
          const response = await axios.post('http://localhost:8080/customQuery', requestBody);
          console.log("SUCCESS!")
          //console.log(response)
          setRows(response.data);
  
        } catch (error) {
          console.log("FAIL!")
          console.error('custom query error:', error.response);
        }

    };


    const handleChange = (event) => {
      setGene(event.target.value);
    };



  const ManageHpo = function ({ fileId , hpoList, setHpoList}) {
    
      return <Tags title={<span style={{ color: 'black' }}>Symptoms</span>} options={HPO_OPTIONS} value={hpoList} onChange={setHpoList} />
    }


    

  const [hpoList, setHpoList] = useHpo({ fileId })
  return (
      <Page style={{
          position: 'absolute', 
          top: 0, 
          left: 0, 
          width: '100%', 
          height: '100vh', 
          backgroundImage: 'url("/static/new_images/things2.png")', 
          backgroundSize: 'cover', 
          backgroundPosition: 'center center',
          backgroundColor: 'rgba(255, 255, 255, 0.8)', // Adds white transparency
          backgroundBlendMode: 'overlay' // This blends the background color with the image
        }}>

  <Box p={3} mt={4} >
  <Typography variant="h5" sx={{mt:4}}>Search Population</Typography>
    <Grid container spacing={2} alignItems="flex-end" mt={4}>
      <Grid item xs={6}>
          <ManageHpo fileId={fileId} sampleName={sampleName} hpoList={hpoList} setHpoList={setHpoList}  />
      </Grid>
      <Grid item container xs={12} sm={6} spacing={2}>
        <Grid item xs={6}>
          <TextField
            fullWidth
            label="Age Interval Start"
            type="number"
            value={ageIntervalStart}
            onChange={(e) => setAgeIntervalStart(e.target.value)}
            InputLabelProps={{
              style: { color: 'black' }
            }}
            InputProps={{
              style: { color: 'black' },
            }}
          />
        </Grid>
        <Grid item xs={6}>
          <TextField
            fullWidth
            label="Age Interval End"
            type="number"
            value={ageIntervalEnd}
            onChange={(e) => setAgeIntervalEnd(e.target.value)}
            InputLabelProps={{
              style: { color: 'black' }
            }}
            InputProps={{
              style: { color: 'black' },
            }}
          />
        </Grid>
      </Grid>

      <Grid item xs={6}>
          <FormControl fullWidth>
              <InputLabel style={{ color: 'black' }} >Gene Specification</InputLabel>
              <Select
              multiple
              value={gene}
              onChange={handleChange}
              variant="outlined"
              label='Gene Specification'
              renderValue={(selected) => selected.join(', ')}
              style={{ borderColor: 'black', color: 'black' }}
              >
              {sortedGeneOptions.map((option) => (
                <MenuItem key={option} value={option}>
                  {option}
                </MenuItem>
              ))}

              </Select>
          </FormControl>
      </Grid>
      <Grid item container xs={6} spacing={2} alignItems="center" >
        <Grid item xs={6}>
        <FormControl fullWidth>
          <InputLabel style={{ color: 'black' }} id="gender-select-label">Gender</InputLabel>
          <Select
            labelId="gender-select-label"
            id="gender-select"
            value={gender}
            label="Gender"
            onChange={(e) => setGender(e.target.value)}
          >
            <MenuItem value="Male">Male</MenuItem>
            <MenuItem value="Female">Female</MenuItem>
          </Select>
        </FormControl>
      </Grid>
    <Grid item>
      <Button variant="contained" color="primary" onClick={handleSearch}>
        Search
      </Button>
      </Grid>
      </Grid>
      </Grid>
      <Box mt={12}>
    </Box>
  </Box>

  <Box p={3} mt={4}>
  {rows.length > 0 ? (
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Name</TableCell>
              <TableCell align="right">Age</TableCell>
              <TableCell align="right">Sex</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.map((row) => (
              <TableRow key={row.id}>
                <TableCell component="th" scope="row">
                  {row.name}
                </TableCell>
                <TableCell align="right">{row.age}</TableCell>
                <TableCell align="right">{row.sex}</TableCell>
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




      

      </Page>
  )

}

export default CustomQueryTable
