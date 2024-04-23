import React from 'react';
import { Box, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography } from '@mui/material';
import { useParams } from 'react-router-dom';

const NewVariantDashboardTable = () => {
    const { fileName } = useParams();
    const data = [
        { variant: "ACMG: PM1, Strength: Pathogenic", diseases: "Alzheimer's Disease, Dementia", geneSymbol: "APOE", gt: "het", frequency: 0.02, priovarScore: 0.95 },
        { variant: "ACMG: PS3, Strength: VUS", diseases: "Breast Cancer", geneSymbol: "BRCA2", gt: "het", frequency: 0.03, priovarScore: 0.7 },
        { variant: "ACMG: PM3, Strength: Benign", diseases: "Lynch Syndrome", geneSymbol: "MLH1", gt: "hom", frequency: 0.04, priovarScore: 0.1 },
        { variant: "ACMG: PM1, Strength: Likely Benign", diseases: "Marfan Syndrome, Lynch Syndrome", geneSymbol: "MECP2", gt: "hom", frequency: 0.04, priovarScore: 0.1 },
        { variant: "ACMG: PM4, Strength: Pathogenic", diseases: "Cystic Fibrosis, Marfan Syndrome, Tay-Sachs Disease", geneSymbol: "CFTR", gt: "het", frequency: 0.01, priovarScore: 0.9 },
        { variant: "ACMG: PM2, Strength: Likely benign", diseases: "Hemochromatosis", geneSymbol: "HFE", gt: "hom", frequency: 0.05, priovarScore: 0.1 },
        { variant: "ACMG: PM6, Strength: VUS", diseases: "Huntington's Disease", geneSymbol: "HTT", gt: "het", frequency: 0.06, priovarScore: 0.2 },
        { variant: "ACMG: PS1, Strength: Benign", diseases: "Parkinson's Disease, Amyotrophic Lateral Sclerosis", geneSymbol: "LRRK2", gt: "hom", frequency: 0.07, priovarScore: 0.3 },
        { variant: "ACMG: PS2, Strength: Likely Benign", diseases: "Amyotrophic Lateral Sclerosis, Marfan Syndrome, Sickle Cell Disease", geneSymbol: "HBB", gt: "het", frequency: 0.08, priovarScore: 0.4 },
        { variant: "ACMG: PS2, Strength: Pathogenic", diseases: "Marfan Syndrome", geneSymbol: "FBN1", gt: "het", frequency: 0.28, priovarScore: 0.83 },
        { variant: "ACMG: PM2, Strength: Pathogenic", diseases: "Alzheimer's Disease", geneSymbol: "APOE", gt: "hom", frequency: 0.08, priovarScore: 0.75 },
        { variant: "ACMG: PS3, Strength: Likely Pathogenic", diseases: "Wilson Disease, Sickle Cell Disease", geneSymbol: "ATP7B", gt: "het", frequency: 0.09, priovarScore: 0.5 },
        { variant: "ACMG: PS4, Strength: Likely benign", diseases: "Charcot-Marie-Tooth Disease", geneSymbol: "PMP22", gt: "hom", frequency: 0.1, priovarScore: 0.25 },
        { variant: "ACMG: BP1, Strength: VUS", diseases: "Amyotrophic Lateral Sclerosis", geneSymbol: "SOD1", gt: "het", frequency: 0.11, priovarScore: 0.35 },
        { variant: "ACMG: PM2, Strength: Benign", diseases: "Polycystic Kidney Disease, Dementia, Huntington's Disease", geneSymbol: "PKD2", gt: "hom", frequency: 0.12, priovarScore: 0.15 },
        { variant: "ACMG: BP3, Strength: Likely Pathogenic", diseases: "Rett Syndrome", geneSymbol: "MECP2", gt: "het", frequency: 0.13, priovarScore: 0.45 },
        { variant: "ACMG: BP4, Strength: Pathogenic", diseases: "Tay-Sachs Disease, Hemochromatosis", geneSymbol: "HEXA", gt: "het", frequency: 0.14, priovarScore: 0.65 },
        { variant: "ACMG: PM5, Strength: Likely benign", diseases: "Lynch Syndrome", geneSymbol: "HFE", gt: "hom", frequency: 0.45, priovarScore: 0.31 },
        { variant: "ACMG: BP5, Strength: Pathogenic", diseases: "Sickle Cell Disease, Huntington's Disease", geneSymbol: "APOE", gt: "het", frequency: 0.15, priovarScore: 0.8 },
        { variant: "ACMG: PS2, Strength: VUS", diseases: "Amyotrophic Lateral Sclerosis, Sickle Cell Disease", geneSymbol: "HBB", gt: "hom", frequency: 0.3, priovarScore: 0.5 },
    ]

    const getHighlightStyle = (score) => {
        if (score < 0.25) return { backgroundColor: '#ccffcc' }; // green
        if (score < 0.75) return { backgroundColor: '#ffff99' }; // yellow
        return { backgroundColor: '#ffcccc' }; // red
    };

    return (
        <TableContainer component={Paper}>
            <Typography variant="h6" sx={{ padding: 2 , mt:5}}>
                All detected variants of '{fileName}'
            </Typography>
            <Table sx={{ padding: 2 , mt:5}}>
                <TableHead>
                    <TableRow>
                        <TableCell>Variant</TableCell>
                        <TableCell>Related diseases</TableCell>
                        <TableCell>Gene symbol</TableCell>
                        <TableCell>GT</TableCell>
                        <TableCell>Local Frequency</TableCell>
                        <TableCell>Priovar Score</TableCell>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {data.map((row, index) => (
                        <TableRow key={index} sx={{
                            '&:hover': {
                                backgroundColor: '#f5f5f5' // hover highlight
                            }
                        }}>
                            <TableCell>{row.variant}</TableCell>
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
    );
};

export default NewVariantDashboardTable;
