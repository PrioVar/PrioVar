import React from 'react'
import Page from 'src/components/Page'
import { /*Container,*/ Grid } from '@material-ui/core'

import AddNewClinician from './AddClinician'


// TODO: UploadVCF and UploadFastq should be abstracted into a single component
// Likewise for FilesTable

const AddClinician = function () {
  return (
    <Page title="Add A New Clinician">
      <Grid container spacing={5}>

        <Grid item xs={12}>
          <AddNewClinician />
        </Grid>
      </Grid>
    </Page>
  )
}

export default AddClinician