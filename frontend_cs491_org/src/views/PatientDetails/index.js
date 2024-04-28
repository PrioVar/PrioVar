import React from 'react'
import Page from 'src/components/Page'
import { Container, Grid } from '@material-ui/core'

import PatientDetailsTable from './PatientDetailsTable'


// TODO: UploadVCF and UploadFastq should be abstracted into a single component
// Likewise for FilesTable

const PatientDetails = function () {
  return (
    <Page title="Patient Details">
      <Grid container spacing={5}>

        <Grid item xs={12}>
          <PatientDetailsTable />
        </Grid>
      </Grid>
    </Page>
  )
}

export default PatientDetails
