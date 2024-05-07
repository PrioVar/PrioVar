import {
  Box, Button, Typography, Grid, TextField, InputLabel, FormControl, Select, MenuItem,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Tooltip, IconButton, Dialog,
  DialogTitle, DialogContent, DialogContentText, DialogActions
} from '@material-ui/core';
import HelpOutlineIcon from '@material-ui/icons/HelpOutline';
import React, { useState } from 'react';
import { useParams } from 'react-router-dom';
import Page from 'src/components/Page';
import Tags from 'src/components/Tags';
import { useHpo } from '../../api/vcf';
import { HPO_OPTIONS } from 'src/constants';

const CustomQueryTable = () => {
  const { fileId } = useParams();
  const [gene, setGene] = useState([]);
  const [ageIntervalStart, setAgeIntervalStart] = useState('');
  const [ageIntervalEnd, setAgeIntervalEnd] = useState('');
  const [gender, setGender] = useState('');
  const [rows, setRows] = useState([]);
  const [isDialogOpen, setIsDialogOpen] = useState(false);

  const handleDialogOpen = () => setIsDialogOpen(true);
  const handleDialogClose = () => setIsDialogOpen(false);

  const geneOptions = ['ABCA1', 'ABCA2', 'ABCA3', 'ABCA4', 'ABCB7', 'ABAT', 'ABL1', 'NAT2', 'AARS1'];
  const sortedGeneOptions = geneOptions.sort((a, b) => a.localeCompare(b));

  const handleChange = (event) => {
    setGene(event.target.value);
  };

  const handleSearch = async () => {
      // Implementation remains the same
  };

  const [hpoList, setHpoList] = useHpo({ fileId });

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
      backgroundColor: 'rgba(255, 255, 255, 0.8)',
      backgroundBlendMode: 'overlay'
    }}>
      <Box p={3} mt={4} display="flex" justifyContent="space-between">
        <Typography variant="h5">Search Population</Typography>
        <Tooltip title="Click here for more information on how to use this page.">
          <IconButton onClick={handleDialogOpen}>
            <HelpOutlineIcon />
          </IconButton>
        </Tooltip>
      </Box>
      
      {/* Dialog component to show the description */}
      <Dialog open={isDialogOpen} onClose={handleDialogClose} aria-labelledby="description-dialog-title">
        <DialogTitle id="description-dialog-title">Page Description</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Welcome to the Population Search Tab! 
            <br /><br />
            Here you can search for patients in the database based on their genetic and phenotypic information.Priovar will use the information you provide to search for patients that match the criteria you specify.
            <br />
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDialogClose} color="primary">Close</Button>
        </DialogActions>
      </Dialog>

      <Grid container spacing={0} alignItems="flex-end" mt={4}>
        <Grid item xs={6}>
          <Tags title={<span style={{ color: 'black' }}>Symptoms</span>} options={HPO_OPTIONS} value={hpoList} onChange={setHpoList} />
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
            <InputLabel style={{ color: 'black' }}>Gene Specification</InputLabel>
            <Select
              multiple
              value={gene}
              onChange={handleChange}
              variant="outlined"
              label="Gene Specification"
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
        <Grid item container xs={6} spacing={2} alignItems="center">
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
      <Box mt={12} />
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
  );
};

export default CustomQueryTable;
