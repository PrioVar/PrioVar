import React from 'react'
import Page from 'src/components/Page'
import { Container, Grid } from '@material-ui/core'

import SimilarPatientsTable from './SimilarPatientsTable'


// TODO: UploadVCF and UploadFastq should be abstracted into a single component
// Likewise for FilesTable

const SimilarPatients = function () {
  return (
    <Page title="Searching Similar Patients">
      <Grid container spacing={5}>

        <Grid item xs={12}>
          <SimilarPatientsTable />
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

export default SimilarPatients
