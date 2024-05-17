
import { CircularProgress } from '@material-ui/core';
import React, { useState, useMemo, useEffect } from 'react';
import { Box, Button, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography, IconButton, Dialog, DialogActions, DialogContent, DialogTitle, TextField, FormGroup, FormControlLabel, Checkbox, RadioGroup, Radio } from '@mui/material';
import { useParams, useNavigate } from 'react-router-dom';
import { sortRows, filterRows } from './tableUtils.js'; // You need to create this utility file for sorting and filtering logic
import ArrowBack from '@mui/icons-material/ArrowBack';
import SortIcon from '@mui/icons-material/Sort';
import axios from 'axios';
import { fetchPatientVariants } from '../../api/file';

const NewVariantDashboardTable = () => {
    const { fileName } = useParams();
    const navigate = useNavigate();
    const [sortConfig, setSortConfig] = useState({ key: null, direction: 'ascending' });
    const [filterConfig, setFilterConfig] = useState({ gts: [], freqRange: [0, 1], scoreRange: [0, 1], strengths: [] });
    const [filterOpen, setFilterOpen] = useState(false);
    const [termsOpen, setTermsOpen] = useState(false);
    const [phenotypeTerms, setPhenotypeTerms] = useState([]);
    const [dataRetriever, setDataRetriever] = useState([]);
    const patientId = localStorage.getItem('patientId');
    var [data, setData] = useState([]);
    
    var chromData = useMemo(() => [
        { variantPosition: "chr3:98456231" },
        { variantPosition: "chr2:29465721",  },
        { variantPosition: "chr2:29465721", },
        { variantPosition: "chr3:98456231", },
        { variantPosition: "chr16:8654329", },
        { variantPosition: "chr3:98456231", },
        { variantPosition: "chr1:154073546", },
        { variantPosition: "chr2:29465721", },
        { variantPosition: "chr1:154073546",},
        { variantPosition: "chr3:98456231", },
    ], []);
    
    
    function setAcmgScoreFromPriovar(data) {
        const acmgScoreMap = {
            "1.00-0.80": "Pathogenic",
            "0.80-0.60": "Likely Pathogenic",
            "0.60-0.40": "VUS",
            "0.40-0.20": "Likely Benign",
            "0.20-0.00": "Benign"
            // Add more ranges as needed
        };
        
        return data.map(entry => {
            let matchedAcmgScore = ''; // Default to original value
    
            // Iterate through each range and find the matching acmgScore
            Object.keys(acmgScoreMap).forEach(range => {
                const [max, min] = range.split("-").map(Number);
                if (entry.priovar_score <= max && entry.priovar_score > min) {
                    
                    matchedAcmgScore = `${acmgScoreMap[range]}`;
                }
            });
    
            return {
                ...entry,
                acmgScore: matchedAcmgScore
            };
        });
        
    }
    data = setAcmgScoreFromPriovar(data);
    
    const fetchVariants = async () => {
        setDataRetriever(false)
        try {
            const response = await fetchPatientVariants(patientId);
            setData(response.data);
            console.log(response.data)
            console.log("hehehehhehdata")
            console.log(data);
            console.log("njsjsdasajhhjhgasdjhg")
            
        } catch (error) {
            console.error('Failed to fetch patient variants', error);
        }
        setDataRetriever(true)
    }

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await axios.get(`http://localhost:8080/patient/termsByFileName/${fileName}`);
                console.log("response.data: ", response.data)
                console.log("response.data: ", response.data)
                console.log("response.data: ", response.data)
                setPhenotypeTerms(response.data); // assuming the response is the list of strings directly
                
            } catch (error) {
                console.error('Failed to fetch phenotype terms', error);
            }
        };

        fetchVariants();
        /*
        if (termsOpen) {
            fetchData();
        }
        */
        fetchData()
    }, []);


    const handleOpenTerms = () => {
        setTermsOpen(true);
    };

    const handleCloseTerms = () => {
        setTermsOpen(false);
    };

    
    const sortedData = useMemo(() => {
        let sortableData = filterRows(data, filterConfig);
        if (sortConfig.key) {
            sortableData = sortRows(sortableData, sortConfig.key, sortConfig.direction);
        }
        return sortableData;
    }, [data, sortConfig, filterConfig]);
    
    const handleSort = (key) => {
        setSortConfig({
            key,
            direction: sortConfig.direction === 'ascending' ? 'descending' : 'ascending'
        });
    };

    const handleStrengthChange = (event, strength) => {
        setFilterConfig(prev => {
            const newStrengths = [...prev.strengths];
            if (event.target.checked) {
                newStrengths.push(strength);
            } else {
                const index = newStrengths.indexOf(strength);
                if (index > -1) {
                    newStrengths.splice(index, 1);
                }
            }
            return { ...prev, strengths: newStrengths };
        });
    };

    const handleChangeCheckbox = (event, gt) => {
        setFilterConfig(prev => {
            const newGts = [...prev.gts];
            if (event.target.checked) {
                newGts.push(gt);
            } else {
                const index = newGts.indexOf(gt);
                if (index > -1) {
                    newGts.splice(index, 1);
                }
            }
            return { ...prev, gts: newGts };
        });
    };

    const handleFilterChange = (event) => {
        const { name, value } = event.target;
        setFilterConfig(prev => ({
            ...prev,
            [name]: prev[name] === value ? '' : value  // Toggle the selection
        }));
    };
    
    

    const handleRangeChange = (event, key, minOrMax) => {
        const value = parseFloat(event.target.value); // use parseFloat to get floating point numbers
        setFilterConfig(prev => {
            const range = [...prev[key]];
            if (minOrMax === 'min') {
                range[0] = value;
            } else {
                range[1] = value;
            }
            return { ...prev, [key]: range };
        });
    };
    
    

    const openFilter = () => {
        setFilterOpen(true);
    };

    const closeFilter = () => {
        setFilterOpen(false);
    };

    const applyFilters = () => {
        closeFilter();
    };

    const resetFilters = () => {
        setFilterConfig({
            gts: [],
            freqRange: [0, 1],
            scoreRange: [0, 1],
            strengths: []
        });
    };    

    const getHighlightStyle = (score) => {
        if (score < 0.25) return { backgroundColor: '#ccffcc' }; // green
        if (score < 0.75) return { backgroundColor: '#ffff99' }; // yellow
        return { backgroundColor: '#ffcccc' }; // red
    };

    return (
        <>
            <Button onClick={() => navigate(-1)} sx={{ ml:1, mt: 3 }}>
                <ArrowBack sx={{ mr: 1 }} /> Go back to files
            </Button>
            <Dialog open={filterOpen} onClose={closeFilter}>
                <DialogTitle>Filter Options</DialogTitle>
                <DialogContent>
                    {/*
                    <Typography sx={{ mt: 2 }} variant="subtitle1" gutterBottom>Local Frequency</Typography>
                    <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                        <TextField
                            label="Min"
                            type="number"
                            inputProps={{ step: "0.01" }}  // allows decimal values up to 2 decimal places
                            value={filterConfig.freqRange[0]}
                            onChange={(e) => handleRangeChange(e, 'freqRange', 'min')}
                        />
                        <TextField
                            label="Max"
                            type="number"
                            inputProps={{ step: "0.01" }}  // allows decimal values up to 2 decimal places
                            value={filterConfig.freqRange[1]}
                            onChange={(e) => handleRangeChange(e, 'freqRange', 'max')}
                        />
                    </Box>
                    */}
                    <Typography variant="subtitle1" gutterBottom>Priovar Score</Typography>
                    <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                        <TextField
                            label="Min"
                            type="number"
                            inputProps={{ step: "0.01" }}  // allows decimal values up to 2 decimal places
                            value={filterConfig.scoreRange[0]}
                            onChange={(e) => handleRangeChange(e, 'scoreRange', 'min')}
                        />
                        <TextField
                            label="Max"
                            type="number"
                            inputProps={{ step: "0.01" }}  // allows decimal values up to 2 decimal places
                            value={filterConfig.scoreRange[1]}
                            onChange={(e) => handleRangeChange(e, 'scoreRange', 'max')}
                        />
                    </Box>
                    {/*
                    <Typography variant="subtitle1" gutterBottom>GT</Typography>
                    <FormGroup>
                        {['het', 'hom'].map(gt => (
                            <FormControlLabel
                                key={gt}
                                control={
                                    <Checkbox
                                        checked={filterConfig.gts.includes(gt)}
                                        onChange={(e) => handleChangeCheckbox(e, gt)}
                                        name="gts"
                                    />
                                }
                                label={gt}
                            />
                        ))}
                    </FormGroup>
                    */}
                    <Typography sx={{ mt: 2 }} variant="subtitle1" gutterBottom>ACMG Strength</Typography>
                    <FormGroup>
                        {['Benign', 'Likely Benign', 'VUS', 'Likely Pathogenic', 'Pathogenic'].map(strength => (
                            <FormControlLabel
                                key={strength}
                                control={
                                    <Checkbox
                                        checked={filterConfig.strengths.includes(strength)}
                                        onChange={(e) => handleStrengthChange(e, strength)}
                                        name="strengths"
                                    />
                                }
                                label={strength}
                            />
                        ))}
                    </FormGroup>
                </DialogContent>
                <DialogActions>
                    <Button onClick={resetFilters}>Reset</Button>
                    <Button onClick={applyFilters}>Done</Button>
                </DialogActions>
            </Dialog>

            {!dataRetriever ? (<CircularProgress/>) : (
                <TableContainer component={Paper}>
                <Typography variant="h6" sx={{  mt: 4, ml:2 }}>
                    All detected variants of '{fileName}'
                </Typography>
                <Button onClick={openFilter} sx={{ float: 'right' }}>Filter</Button>

                <Button onClick={handleOpenTerms} sx={{ float: 'right', mr: 2 }}>See Phenotype Terms</Button>
                <Dialog open={termsOpen} onClose={handleCloseTerms}>
                    <DialogTitle> {phenotypeTerms.length} Phenotype Terms</DialogTitle>
                    <DialogContent sx={{mt:2}}>
                        <Typography variant="body1" component="div">
                            {phenotypeTerms.length > 0 ? (
                                <ul>
                                    {phenotypeTerms.map((term, index) => (
                                        <li key={index}>{term}</li>
                                    ))}
                                </ul>
                            ) : (
                                <Typography variant="subtitle1">No terms available</Typography>
                            )}
                        </Typography>
                    </DialogContent>
                    <DialogActions>
                        <Button onClick={handleCloseTerms}>Close</Button>
                    </DialogActions>
                </Dialog>
                
                <Table sx={{ padding: 2, mt: 5 }}>
                    <TableHead>
                        <TableRow>
                            <TableCell>Chrom</TableCell>
                            <TableCell>Pos</TableCell>
                            <TableCell>Ref</TableCell>
                            <TableCell>Alt</TableCell>
                            <TableCell>Qual</TableCell>
                            <TableCell>Allele</TableCell>
                            <TableCell onClick={() => handleSort('variantPosition')} >Consequence</TableCell>
                            <TableCell onClick={() => handleSort('acmgScore')}>ACMG Score</TableCell>
                            <TableCell>Symbol</TableCell>
                            <TableCell>Gene</TableCell>
                            <TableCell>HGSVC</TableCell>
                            <TableCell>Turkish Variome</TableCell>
                            <TableCell>Alphamissense Score</TableCell>
                            {
                                /*
                                <TableCell>
                                    <IconButton onClick={() => handleSort('frequency')}>
                                        <SortIcon />
                                    </IconButton>
                                    Local Frequency
                                </TableCell>
                                */
                            }
                            
                            <TableCell>
                                <IconButton onClick={() => handleSort('priovar_score')}>
                                    <SortIcon />
                                </IconButton>
                                Priovar Score
                            </TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {sortedData.map((row, index) => (
                            <TableRow key={index} sx={{
                                '&:hover': {
                                    backgroundColor: '#f5f5f5' // hover highlight
                                }
                            }}>
                                <TableCell style={{ maxWidth: '50px', whiteSpace: 'normal', wordWrap: 'break-word' }}>{row.chrom}</TableCell>
                                <TableCell style={{ maxWidth: '80px', whiteSpace: 'normal', wordWrap: 'break-word' }}>{row.pos}</TableCell>
                                <TableCell style={{ maxWidth: '50px', whiteSpace: 'normal', wordWrap: 'break-word' }}>{row.ref}</TableCell>
                                <TableCell style={{ maxWidth: '50px', whiteSpace: 'normal', wordWrap: 'break-word' }}>{row.alt}</TableCell>
                                <TableCell>{row.qual}</TableCell>
                                <TableCell>{row.allele}</TableCell>
                                <TableCell style={{ maxWidth: '200px', whiteSpace: 'normal', wordWrap: 'break-word' }}>
                                    {row.consequence}
                                </TableCell>
                                <TableCell>{row.acmgScore}</TableCell>
                                <TableCell>{row.symbol}</TableCell>
                                <TableCell>{row.gene}</TableCell>
                                <TableCell style={{ maxWidth: '200px', whiteSpace: 'normal', wordWrap: 'break-word' }}>
                                    {row.hgsvc_original}
                                </TableCell>
                                <TableCell>{row.turkishvariome_tv_af_original}</TableCell>
                                <TableCell>{row.alpha_missense_score_mean}</TableCell>
                                {
                                    /* 
                                    <TableCell>{row.diseases}</TableCell>
                                    <TableCell>{row.geneSymbol}</TableCell>
                                    <TableCell>{row.gt}</TableCell>
                                    <TableCell>{row.frequency.toFixed(2)}</TableCell>
                                    */
                                }
                                <TableCell align="left">
                                    <Box sx={{
                                        display: 'inline-flex',
                                        alignItems: 'center',
                                        p: 0.5,
                                        borderRadius: 1,
                                        ...getHighlightStyle(row.priovar_score)
                                    }}>
                                        {row.priovar_score.toFixed(2)}
                                    </Box>
                                </TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>
            )}
            
        </>
    );
};

export default NewVariantDashboardTable;

/*
import React, { useState, useMemo, useEffect } from 'react';
import { Box, Button, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography, IconButton, Dialog, DialogActions, DialogContent, DialogTitle, TextField, FormGroup, FormControlLabel, Checkbox, RadioGroup, Radio } from '@mui/material';
import { useParams, useNavigate } from 'react-router-dom';
import { sortRows, filterRows } from './tableUtils.js'; // You need to create this utility file for sorting and filtering logic
import ArrowBack from '@mui/icons-material/ArrowBack';
import SortIcon from '@mui/icons-material/Sort';
import axios from 'axios';
import { fetchPatientVariants } from '../../api/file';


const NewVariantDashboardTable = () => {
    const { fileName } = useParams();
    const navigate = useNavigate();
    const [sortConfig, setSortConfig] = useState({ key: null, direction: 'ascending' });
    const [filterConfig, setFilterConfig] = useState({ gts: [], freqRange: [0, 1], scoreRange: [0, 1], strengths: [] });
    const [filterOpen, setFilterOpen] = useState(false);
    const [termsOpen, setTermsOpen] = useState(false);
    const [phenotypeTerms, setPhenotypeTerms] = useState([]);
    const patientId = localStorage.getItem('patientId');
    var data = useMemo(() => [
        { acmgScore: "ACMG: PM1, Strength: Pathogenic", variantPosition: "chr16:8654329", diseases: "Alzheimer's Disease, Dementia", geneSymbol: "APOE", gt: "het", frequency: 0.02, priovarScore: 0.95 },
        { acmgScore: "ACMG: PS3, Strength: VUS", variantPosition: "chr3:98456231", diseases: "Breast Cancer", geneSymbol: "BRCA2", gt: "het", frequency: 0.03, priovarScore: 0.7 },
        { acmgScore: "ACMG: PM3, Strength: Benign", variantPosition: "chr1:154073546", diseases: "Lynch Syndrome", geneSymbol: "MLH1", gt: "hom", frequency: 0.04, priovarScore: 0.1 },
        { acmgScore: "ACMG: PM1, Strength: Likely Benign", variantPosition: "chr16:8654329", diseases: "Marfan Syndrome, Lynch Syndrome", geneSymbol: "MECP2", gt: "hom", frequency: 0.04, priovarScore: 0.1 },
        { acmgScore: "ACMG: PM4, Strength: Pathogenic", variantPosition: "chr1:154073546", diseases: "Cystic Fibrosis, Marfan Syndrome, Tay-Sachs Disease", geneSymbol: "CFTR", gt: "het", frequency: 0.01, priovarScore: 0.9 },
        { acmgScore: "ACMG: PM2, Strength: Likely benign", variantPosition: "chr2:29465721", diseases: "Hemochromatosis", geneSymbol: "HFE", gt: "hom", frequency: 0.05, priovarScore: 0.1 },
        { acmgScore: "ACMG: PM6, Strength: VUS", variantPosition: "chr3:98456231", diseases: "Huntington's Disease", geneSymbol: "HTT", gt: "het", frequency: 0.06, priovarScore: 0.2 },
        { acmgScore: "ACMG: PS1, Strength: Benign", variantPosition: "chr16:8654329", diseases: "Parkinson's Disease, Amyotrophic Lateral Sclerosis", geneSymbol: "LRRK2", gt: "hom", frequency: 0.07, priovarScore: 0.3 },
        { acmgScore: "ACMG: PS2, Strength: Likely Benign", variantPosition: "chr16:8654329", diseases: "Amyotrophic Lateral Sclerosis, Marfan Syndrome, Sickle Cell Disease", geneSymbol: "HBB", gt: "het", frequency: 0.08, priovarScore: 0.4 },
        { acmgScore: "ACMG: PS2, Strength: Pathogenic", variantPosition: "chr1:154073546", diseases: "Marfan Syndrome", geneSymbol: "FBN1", gt: "het", frequency: 0.28, priovarScore: 0.83 },
        { acmgScore: "ACMG: PM2, Strength: Pathogenic", variantPosition: "chr3:98456231", diseases: "Alzheimer's Disease", geneSymbol: "APOE", gt: "hom", frequency: 0.08, priovarScore: 0.75 },
        { acmgScore: "ACMG: PS3, Strength: Likely Pathogenic", variantPosition: "chr2:29465721", diseases: "Wilson Disease, Sickle Cell Disease", geneSymbol: "ATP7B", gt: "het", frequency: 0.09, priovarScore: 0.5 },
        { acmgScore: "ACMG: PS4, Strength: Likely benign", variantPosition: "chr2:29465721", diseases: "Charcot-Marie-Tooth Disease", geneSymbol: "PMP22", gt: "hom", frequency: 0.1, priovarScore: 0.25 },
        { acmgScore: "ACMG: BP1, Strength: VUS", variantPosition: "chr3:98456231", diseases: "Amyotrophic Lateral Sclerosis", geneSymbol: "SOD1", gt: "het", frequency: 0.11, priovarScore: 0.35 },
        { acmgScore: "ACMG: PM2, Strength: Benign", variantPosition: "chr16:8654329", diseases: "Polycystic Kidney Disease, Dementia, Huntington's Disease", geneSymbol: "PKD2", gt: "hom", frequency: 0.12, priovarScore: 0.15 },
        { acmgScore: "ACMG: BP3, Strength: Likely Pathogenic", variantPosition: "chr3:98456231", diseases: "Rett Syndrome", geneSymbol: "MECP2", gt: "het", frequency: 0.13, priovarScore: 0.45 },
        { acmgScore: "ACMG: BP4, Strength: Pathogenic", variantPosition: "chr1:154073546", diseases: "Tay-Sachs Disease, Hemochromatosis", geneSymbol: "HEXA", gt: "het", frequency: 0.14, priovarScore: 0.65 },
        { acmgScore: "ACMG: PM5, Strength: Likely benign", variantPosition: "chr2:29465721", diseases: "Lynch Syndrome", geneSymbol: "HFE", gt: "hom", frequency: 0.45, priovarScore: 0.31 },
        { acmgScore: "ACMG: BP5, Strength: Pathogenic", variantPosition: "chr1:154073546", diseases: "Sickle Cell Disease, Huntington's Disease", geneSymbol: "APOE", gt: "het", frequency: 0.15, priovarScore: 0.8 },
        { acmgScore: "ACMG: PS2, Strength: VUS", variantPosition: "chr3:98456231", diseases: "Amyotrophic Lateral Sclerosis, Sickle Cell Disease", geneSymbol: "HBB", gt: "hom", frequency: 0.3, priovarScore: 0.5 },
    ], []);
    
    const newData = fetchPatientVariants(patientId);
    console.log("hehehehheh")
    console.log(newData);
    console.log("njsjsdasajhhjhgasdjhg")
    function setAcmgScoreFromPriovar(data) {
        const acmgScoreMap = {
            "1.00-0.80": "Pathogenic",
            "0.80-0.60": "Likely Pathogenic",
            "0.60-0.40": "VUS",
            "0.40-0.20": "Likely Benign",
            "0.20-0.00": "Benign"
            // Add more ranges as needed
        };
    
        return data.map(entry => {
            let matchedAcmgScore = entry.acmgScore; // Default to original value
    
            // Iterate through each range and find the matching acmgScore
            Object.keys(acmgScoreMap).forEach(range => {
                const [max, min] = range.split("-").map(Number);
                if (entry.priovarScore <= max && entry.priovarScore > min) {
                    matchedAcmgScore = `ACMG Strength: ${acmgScoreMap[range]}`;
                }
            });
    
            return {
                ...entry,
                acmgScore: matchedAcmgScore
            };
        });
    }
    data = setAcmgScoreFromPriovar(data);
    
    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await axios.get(`http://localhost:8080/patient/termsByFileName/${fileName}`);
                setPhenotypeTerms(response.data); // assuming the response is the list of strings directly
                
            } catch (error) {
                console.error('Failed to fetch phenotype terms', error);
            }
        };

        if (termsOpen) {
            fetchData();
        }
    }, [termsOpen, fileName]);


    const handleOpenTerms = () => {
        setTermsOpen(true);
    };

    const handleCloseTerms = () => {
        setTermsOpen(false);
    };

    const sortedData = useMemo(() => {
        let sortableData = filterRows(data, filterConfig);
        if (sortConfig.key) {
            sortableData = sortRows(sortableData, sortConfig.key, sortConfig.direction);
        }
        return sortableData;
    }, [data, sortConfig, filterConfig]);

    const handleSort = (key) => {
        setSortConfig({
            key,
            direction: sortConfig.direction === 'ascending' ? 'descending' : 'ascending'
        });
    };

    const handleStrengthChange = (event, strength) => {
        setFilterConfig(prev => {
            const newStrengths = [...prev.strengths];
            if (event.target.checked) {
                newStrengths.push(strength);
            } else {
                const index = newStrengths.indexOf(strength);
                if (index > -1) {
                    newStrengths.splice(index, 1);
                }
            }
            return { ...prev, strengths: newStrengths };
        });
    };

    const handleChangeCheckbox = (event, gt) => {
        setFilterConfig(prev => {
            const newGts = [...prev.gts];
            if (event.target.checked) {
                newGts.push(gt);
            } else {
                const index = newGts.indexOf(gt);
                if (index > -1) {
                    newGts.splice(index, 1);
                }
            }
            return { ...prev, gts: newGts };
        });
    };

    const handleFilterChange = (event) => {
        const { name, value } = event.target;
        setFilterConfig(prev => ({
            ...prev,
            [name]: prev[name] === value ? '' : value  // Toggle the selection
        }));
    };
    
    

    const handleRangeChange = (event, key, minOrMax) => {
        const value = parseFloat(event.target.value); // use parseFloat to get floating point numbers
        setFilterConfig(prev => {
            const range = [...prev[key]];
            if (minOrMax === 'min') {
                range[0] = value;
            } else {
                range[1] = value;
            }
            return { ...prev, [key]: range };
        });
    };
    
    

    const openFilter = () => {
        setFilterOpen(true);
    };

    const closeFilter = () => {
        setFilterOpen(false);
    };

    const applyFilters = () => {
        closeFilter();
    };

    const resetFilters = () => {
        setFilterConfig({
            gts: [],
            freqRange: [0, 1],
            scoreRange: [0, 1],
            strengths: []
        });
    };    

    const getHighlightStyle = (score) => {
        if (score < 0.25) return { backgroundColor: '#ccffcc' }; // green
        if (score < 0.75) return { backgroundColor: '#ffff99' }; // yellow
        return { backgroundColor: '#ffcccc' }; // red
    };

    return (
        <>
            <Button onClick={() => navigate(-1)} sx={{ ml:1, mt: 3 }}>
                <ArrowBack sx={{ mr: 1 }} /> Go back to files
            </Button>
            <Dialog open={filterOpen} onClose={closeFilter}>
                <DialogTitle>Filter Options</DialogTitle>
                <DialogContent>
                    <Typography sx={{ mt: 2 }} variant="subtitle1" gutterBottom>Local Frequency</Typography>
                    <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                        <TextField
                            label="Min"
                            type="number"
                            inputProps={{ step: "0.01" }}  // allows decimal values up to 2 decimal places
                            value={filterConfig.freqRange[0]}
                            onChange={(e) => handleRangeChange(e, 'freqRange', 'min')}
                        />
                        <TextField
                            label="Max"
                            type="number"
                            inputProps={{ step: "0.01" }}  // allows decimal values up to 2 decimal places
                            value={filterConfig.freqRange[1]}
                            onChange={(e) => handleRangeChange(e, 'freqRange', 'max')}
                        />
                    </Box>
                    <Typography variant="subtitle1" gutterBottom>Priovar Score</Typography>
                    <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                        <TextField
                            label="Min"
                            type="number"
                            inputProps={{ step: "0.01" }}  // allows decimal values up to 2 decimal places
                            value={filterConfig.scoreRange[0]}
                            onChange={(e) => handleRangeChange(e, 'scoreRange', 'min')}
                        />
                        <TextField
                            label="Max"
                            type="number"
                            inputProps={{ step: "0.01" }}  // allows decimal values up to 2 decimal places
                            value={filterConfig.scoreRange[1]}
                            onChange={(e) => handleRangeChange(e, 'scoreRange', 'max')}
                        />
                    </Box>
                    <Typography variant="subtitle1" gutterBottom>GT</Typography>
                    <FormGroup>
                        {['het', 'hom'].map(gt => (
                            <FormControlLabel
                                key={gt}
                                control={
                                    <Checkbox
                                        checked={filterConfig.gts.includes(gt)}
                                        onChange={(e) => handleChangeCheckbox(e, gt)}
                                        name="gts"
                                    />
                                }
                                label={gt}
                            />
                        ))}
                    </FormGroup>

                    <Typography sx={{ mt: 2 }} variant="subtitle1" gutterBottom>ACMG Strength</Typography>
                    <FormGroup>
                        {['Benign', 'Likely Benign', 'VUS', 'Likely Pathogenic', 'Pathogenic'].map(strength => (
                            <FormControlLabel
                                key={strength}
                                control={
                                    <Checkbox
                                        checked={filterConfig.strengths.includes(strength)}
                                        onChange={(e) => handleStrengthChange(e, strength)}
                                        name="strengths"
                                    />
                                }
                                label={strength}
                            />
                        ))}
                    </FormGroup>
                </DialogContent>
                <DialogActions>
                    <Button onClick={resetFilters}>Reset</Button>
                    <Button onClick={applyFilters}>Done</Button>
                </DialogActions>
            </Dialog>


            <TableContainer component={Paper}>
                <Typography variant="h6" sx={{  mt: 4, ml:2 }}>
                    All detected variants of '{fileName}'
                </Typography>
                <Button onClick={openFilter} sx={{ float: 'right' }}>Filter</Button>

                <Button onClick={handleOpenTerms} sx={{ float: 'right', mr: 2 }}>See Phenotype Terms</Button>
                <Dialog open={termsOpen} onClose={handleCloseTerms}>
                    <DialogTitle> {phenotypeTerms.length} Phenotype Terms</DialogTitle>
                    <DialogContent sx={{mt:2}}>
                        <Typography variant="body1" component="div">
                            {phenotypeTerms.length > 0 ? (
                                <ul>
                                    {phenotypeTerms.map((term, index) => (
                                        <li key={index}>{term}</li>
                                    ))}
                                </ul>
                            ) : (
                                <Typography variant="subtitle1">No terms available</Typography>
                            )}
                        </Typography>
                    </DialogContent>
                    <DialogActions>
                        <Button onClick={handleCloseTerms}>Close</Button>
                    </DialogActions>
                </Dialog>

                <Table sx={{ padding: 2, mt: 5 }}>
                    <TableHead>
                        <TableRow>
                            <TableCell onClick={() => handleSort('variantPosition')}>Variant Position</TableCell>
                            <TableCell onClick={() => handleSort('acmgScore')}>ACMG Score</TableCell>
                            <TableCell>Known related diseases</TableCell>
                            <TableCell>Gene symbol</TableCell>
                            <TableCell>GT</TableCell>
                            <TableCell>
                                <IconButton onClick={() => handleSort('frequency')}>
                                    <SortIcon />
                                </IconButton>
                                Local Frequency
                            </TableCell>
                            <TableCell>
                                <IconButton onClick={() => handleSort('priovarScore')}>
                                    <SortIcon />
                                </IconButton>
                                Priovar Score
                            </TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {sortedData.map((row, index) => (
                            <TableRow key={index} sx={{
                                '&:hover': {
                                    backgroundColor: '#f5f5f5' // hover highlight
                                }
                            }}>
                                <TableCell>{row.variantPosition}</TableCell>
                                <TableCell>{row.acmgScore}</TableCell>
                                <TableCell>{row.diseases}</TableCell>
                                <TableCell>{row.geneSymbol}</TableCell>
                                <TableCell>{row.gt}</TableCell>
                                <TableCell>{row.frequency.toFixed(2)}</TableCell>
                                <TableCell align="left">
                                    <Box sx={{
                                        display: 'inline-flex',
                                        alignItems: 'center',
                                        p: 0.5,
                                        borderRadius: 1,
                                        ...getHighlightStyle(row.priovarScore)
                                    }}>
                                        {row.priovarScore.toFixed(2)}
                                    </Box>
                                </TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>
        </>
    );
};

export default NewVariantDashboardTable;
*/