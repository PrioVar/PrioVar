import React from 'react'
import Page from 'src/components/Page'
import { Container, Grid } from '@material-ui/core'

import MyPatientsTable from './MyPatientsTable'


// TODO: UploadVCF and UploadFastq should be abstracted into a single component
// Likewise for FilesTable

const MyPatients = function () {
  return (
    <Page title="Upload File | Priovar">
      <Grid container spacing={5}>

        <Grid item xs={12}>
          <MyPatientsTable />
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

export default MyPatients
