import {
  Alert,
  Card,
  CardContent,
  CardHeader,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@material-ui/core'
import React from 'react'

const INHERITANCE_MODE_TO_HUMAN_READABLE = {
  AUTOSOMAL_DOMINANT: 'AD',
  AUTOSOMAL_RECESSIVE: 'AR',
  X_DOMINANT: 'XD',
  X_RECESSIVE: 'XR',
  Y_DOMINANT: 'YD',
  Y_RECESSIVE: 'YR',
  UNKNOWN: '?',
}

const OmimTable = function ({ omim }) {
  return (
    <TableContainer>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Phenotypes</TableCell>
            <TableCell>OMIM number</TableCell>
            <TableCell>Inheritance</TableCell>
            <TableCell>Phenotype mapping key</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {omim.phenotypes.map((row) => (
            <TableRow key={row.name} sx={{ '&:last-child td, &:last-child th': { border: 0 } }}>
              <TableCell component="th" scope="row">
                {row.description}
              </TableCell>
              <TableCell>{row.omim_number}</TableCell>
              <TableCell>{INHERITANCE_MODE_TO_HUMAN_READABLE[row.inheritance_mode]}</TableCell>
              <TableCell>{row.mapping_key}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  )
}

const OmimTableContainer = function ({ omim, geneSymbol }) {
  return (
    <Card sx={{ height: '100%' }} component={Stack} direction="column">
      <CardHeader
        title="OMIM"
        titleTypographyProps={{
          variant: 'h4',
          textAlign: 'center',
        }}
      />
      <CardContent
        component={Stack}
        direction="column"
        alignItems={omim ? 'stretch' : 'center'}
        sx={{ flexGrow: 1 }}
        justifyContent="center"
      >
        {omim ? (
          <OmimTable omim={omim} />
        ) : (
          <Alert severity="error" variant="outlined" textAlign="center">
            No OMIM records found for {geneSymbol}
          </Alert>
        )}
      </CardContent>
    </Card>
  )
}

export default OmimTableContainer
