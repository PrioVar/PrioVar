import React from 'react'
import Page from 'src/components/Page'
import { /*Container,*/ Grid } from '@material-ui/core'

import CustomQueryTable from './CustomQuery'


// TODO: UploadVCF and UploadFastq should be abstracted into a single component
// Likewise for FilesTable

const CustomQuery = function () {
  return (
    <Page title="Search Population">
      <Grid container spacing={5}>

        <Grid item xs={12}>
          <CustomQueryTable />
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

export default CustomQuery
