import React from 'react'
import { useEffect, useState } from 'react'
import Page from 'src/components/Page'
import { Container, Grid } from '@material-ui/core'

import FilesTable from './FilesTable'
import UploadVCF from './UploadVCF'
import VariantDashboard from './VariantDashboard'


// TODO: UploadVCF and UploadFastq should be abstracted into a single component
// Likewise for FilesTable

const UploadView = function () {

    const [isFileUploaded, setIsFileUploaded] = useState(false);

    const handleFileUploadComplete = (uploaded) => {
        setIsFileUploaded(uploaded);
    };

  return (
    <Page title="Upload File | Genesus">
      <Grid container spacing={5}>
        <Grid item xs={6}>
          <UploadVCF onUploadComplete={handleFileUploadComplete}/>
        </Grid>

        <Grid item xs={12}>
          <FilesTable />
        </Grid>
      </Grid>
    </Page>
  )
}

export default UploadView
