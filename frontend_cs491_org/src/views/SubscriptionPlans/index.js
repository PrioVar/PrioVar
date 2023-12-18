import React from 'react'
import Page from 'src/components/Page'
import { Container, Grid } from '@material-ui/core'

import SubscriptionPlansTable from './SubscriptionPlansTable'


// TODO: UploadVCF and UploadFastq should be abstracted into a single component
// Likewise for FilesTable

const SubscriptionPlans = function () {
  return (
    <Page title="Subscription Plans">
      <Grid container spacing={5}>

        <Grid item xs={12}>
          <SubscriptionPlansTable />
        </Grid>
      </Grid>
    </Page>
  )
}

export default SubscriptionPlans
