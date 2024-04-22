import React from 'react';
import { Box, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography } from '@mui/material';

const NewVariantDashboardTable = () => {
    const fileName = "example.vcf"; // Example file name
    const data = [
        { variant: "variant1", diseases: "Kidney Failure, Rickets", geneSymbol: "CTNS-AS1", gt: "het", frequency: 0.2, priovarScore: 0.2 },
        { variant: "variant2", diseases: "Diabetes", geneSymbol: "RYR1", gt: "hom", frequency: 0.4, priovarScore: 0.5 },
        { variant: "variant3", diseases: "Heart Disease", geneSymbol: "TTN", gt: "het", frequency: 0.9, priovarScore: 0.8 }
    ];

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
                        <TableCell>Frequency</TableCell>
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
