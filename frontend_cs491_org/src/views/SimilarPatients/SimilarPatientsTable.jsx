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
    MenuItem,
    OutlinedInput, 
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
  

  
  const SimilarPatientsTable = function () {
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
    const [searchType, setSearchType] = useState('');
    const [selectedTerms, setSelectedTerms] = useState([]);;

    //
    const [gene, setGene] = useState('');
    const [ageIntervalStart, setAgeIntervalStart] = useState('');
    const [ageIntervalEnd, setAgeIntervalEnd] = useState('');
    const [gender, setGender] = useState(''); 

    // Dummy values for phenotype and genotype options
    const phenotypeTerms = ['Phenotype Term 1', 'Phenotype Term 2', 'Phenotype Term 3'];
    const genotypeTerms = ['Genotype Term 1', 'Genotype Term 2', 'Genotype Term 3'];

    const handleSearchTypeChange = (event) => {
        setSearchType(event.target.value);
        setSelectedTerms([]); // Reset terms when search type changes
    };

    const handleTermsChange = (event) => {
        setSelectedTerms(event.target.value);
    };

    const handleSearch = async () => {

        console.log(`Searching for ${searchType} with terms`, selectedTerms);

    };

    // Function to render the selected terms as chips
    const renderValue = (selected) => (
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
        {selected.map((value) => (
            <Chip key={value} label={value} />
        ))}
        </Box>
    );

    return (
        <>
    <Box p={3} mt={4}>
    <Typography variant="h5">Search for Similar Patients</Typography>
      <Grid container spacing={2} alignItems="flex-end" mt={4}>
        <Grid item xs={6}>
        <FormControl fullWidth variant="outlined">
            <InputLabel id="search-type-select-label">Search by Phenotype or Genotype</InputLabel>
            <Select
              labelId="search-type-select-label"
              id="search-type-select"
              value={searchType}
              label="Search by Phenotype or Genotype"
              onChange={handleSearchTypeChange}
            >
              <MenuItem value="G">Genotype</MenuItem>
              <MenuItem value="P">Phenotype</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        {searchType && (
            <Grid item xs={6}>
              <FormControl fullWidth variant="outlined">
                <InputLabel id="term-select-label">Choose {searchType === 'P' ? 'phenotype' : 'genotype'} terms</InputLabel>
                <Select
                  labelId="terms-select-label"
                  id="terms-select"
                  multiple
                  value={selectedTerms}
                  onChange={handleTermsChange}
                  input={<OutlinedInput id="select-multiple-chip" 
                  label={`Choose ${searchType === 'P' ? 'phenotype' : 'genotype'} terms`} />}
                  renderValue={renderValue}
                >
                  {searchType === 'P'
                    ? phenotypeTerms.map((term, index) => (
                        <MenuItem key={index} value={term}>
                          {term}
                        </MenuItem>
                      ))
                    : genotypeTerms.map((term, index) => (
                        <MenuItem key={index} value={term}>
                          {term}
                        </MenuItem>
                      ))}
                </Select>
              </FormControl>
            </Grid>
          )}
        <Grid item container xs={12} sm={6} spacing={2}>
          <Grid item xs={6}>
            <Button variant="contained" color="primary" onClick={handleSearch}>
            Search
            </Button>
          </Grid>
        </Grid>


        </Grid>
        <Box mt={12}>
            Results will be displayed here
      </Box>
    </Box>
        </>
    )

  }
  
  export default SimilarPatientsTable
  