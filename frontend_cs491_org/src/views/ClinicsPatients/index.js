import React from 'react'
import Page from 'src/components/Page'
import { Container, Grid } from '@material-ui/core'

import ClinicsPatientsTable from './ClinicsPatientsTable'


// TODO: UploadVCF and UploadFastq should be abstracted into a single component
// Likewise for FilesTable

const ClinicsPatients = function () {
  return (
    <Page title="Clinics Patients | PrioVar">
      <Grid container spacing={5}>

        <Grid item xs={12}>
          <ClinicsPatientsTable />
        </Grid>

        {/*<Grid item xs={12}>*/}
        {/*  <UploadFastq />*/}
        {/*</Grid>*/}

        {/*<Grid item xs={12}>*/}
        {/*  <FastqFilesTable />*/}
        {/*</Grid>*/}
      </Grid>
    </Page>
  )
}

export default ClinicsPatients
