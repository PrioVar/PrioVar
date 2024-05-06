import React, { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
    Box,
    Button,
    Typography,
    Grid,
    CircularProgress,
    IconButton
} from '@material-ui/core';
import { ArrowBack, CloseOutlined } from '@material-ui/icons';
import axios from 'axios';
import { fetchDiseases, fetchPhenotypeTerms } from '../../api/file';
import { useSnackbar } from 'notistack5';
import { ROOTS_PrioVar } from '../../routes/paths';

const ConstPatientDetailsTable = function () {
    const { patientId } = useParams();
    const navigate = useNavigate();
    const { enqueueSnackbar } = useSnackbar();
    const [details, setDetails] = useState({
        name: '', age: '', sex: '', disease: '', assignedClinic: '', phenotypeTerms: []
    });
    const [phenotypeTermsLoading, setPhenotypeTermsLoading] = useState(true);

    const fetchPatientDetails = async () => {
        try {
            const response = await axios.get(`${ROOTS_PrioVar}/patient/${patientId}`);
            setDetails(response.data);
        } catch (error) {
            console.error('Error fetching patient details:', error);
        }
        setPhenotypeTermsLoading(false);
    };

    useEffect(() => {
        fetchPatientDetails();
    }, []);

    const PhenotypeTerm = ({ term }) => (
        <div style={{ border: '1px solid #ccc', borderRadius: '5px', padding: '5px', display: 'inline-flex', alignItems: 'center', marginRight: '5px' }}>
            <Typography variant="body1" style={{ marginRight: '5px' }}>{term.name}</Typography>
            <IconButton size="small" disabled>
                <CloseOutlined color='disabled' />
            </IconButton>
        </div>
    );

    return (
        <>
            <Button onClick={() => navigate(-1)} sx={{ ml: 1, mt: 3 }}>
                <ArrowBack sx={{ mr: 1 }} /> Go Back To Patients
            </Button>
            <Box p={3}>
                <Typography variant="h4" align="center">Patient Details</Typography>
                <Grid container spacing={2} mt={4}>
                    <Grid item xs={3}>
                        <Typography variant="h6">Name:</Typography> {details.name}
                    </Grid>
                    <Grid item xs={3}>
                        <Typography variant="h6">Age:</Typography> {details.age}
                    </Grid>
                    <Grid item xs={3}>
                        <Typography variant="h6">Sex:</Typography> {details.sex}
                    </Grid>
                    <Grid item xs={3}>
                        <Typography variant="h6">Health Center:</Typography> {details.medicalCenter?.name}
                    </Grid>
                    <Grid item xs={4} mt={4}>
                        <Typography variant="h6">Disease:</Typography> {details.disease?.diseaseName || "Not Available"}
                    </Grid>
                    <Grid item xs={8} mt={4}>
                        <Typography variant="h6">Phenotype Terms:</Typography>
                        {phenotypeTermsLoading ? (<CircularProgress />)
                            : (
                                details.phenotypeTerms.map((term, index) => (
                                    <PhenotypeTerm key={index} term={term} />
                                ))
                            )}
                    </Grid>
                </Grid>
            </Box>
        </>
    );
}

export default ConstPatientDetailsTable;
