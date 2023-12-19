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
  
  const CustomQueryTable = function () {
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
    const [gene, setGene] = useState('');
    const [ageIntervalStart, setAgeIntervalStart] = useState('');
    const [ageIntervalEnd, setAgeIntervalEnd] = useState('');
    const [gender, setGender] = useState(''); 

    const handleSearch = async () => {

        const phenotypeTerms = hpoList.map(item => {
            // Extract the numeric part of the HP code and remove leading zeros
            const id = parseInt(item.value.split(':')[1], 10);
            return { id };
        });

        const body = {
            phenotypeTerms: phenotypeTerms,
            geneSpecification: gene,
            sex: gender,
            ageIntervalStart: ageIntervalStart,
            ageIntervalEnd: ageIntervalEnd
        }
        console.log("body:")
        console.log(body)
        try {
            const response = await axios.post('http://localhost:8080/customQuery', body);
            console.log("SUCCESS!")
    
          } catch (error) {
            console.log("FAIL!")
            console.error('custom query error:', error.response);
          }

      };



    const ManageHpo = function ({ fileId , hpoList, setHpoList}) {
      
        return <Tags title="Symptoms" options={HPO_OPTIONS} value={hpoList} onChange={setHpoList} />
      }
  

    const [hpoList, setHpoList] = useHpo({ fileId })
    return (
        <>

    <Box p={3} mt={4}>
    <Typography variant="h5">Search Population</Typography>
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
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Age Interval End"
              type="number"
              value={ageIntervalEnd}
              onChange={(e) => setAgeIntervalEnd(e.target.value)}
            />
          </Grid>
        </Grid>

        <Grid item xs={6}>
          <TextField
            fullWidth
            label="Gene Specification"
            value={gene}
            onChange={(e) => setGene(e.target.value)}
          />
        </Grid>
        <Grid item container xs={6} spacing={2} alignItems="center" >
          <Grid item xs={6}>
          <FormControl fullWidth>
            <InputLabel id="gender-select-label">Gender</InputLabel>
            <Select
              labelId="gender-select-label"
              id="gender-select"
              value={gender}
              label="Gender"
              onChange={(e) => setGender(e.target.value)}
            >
              <MenuItem value="M">Male</MenuItem>
              <MenuItem value="F">Female</MenuItem>
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
            Results will be displayed here
      </Box>
    </Box>




        

        </>
    )

  }
  
  export default CustomQueryTable
  