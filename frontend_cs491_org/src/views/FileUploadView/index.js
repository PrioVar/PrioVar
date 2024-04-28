import React from 'react'
import { useEffect, useState } from 'react'
import Page from 'src/components/Page'
import { Container, Grid, Box } from '@material-ui/core'

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

    const resetFileUploaded = () => {
        setIsFileUploaded(false);
    };

  return (
    <Page title="Upload VCF | PrioVar">
      <Grid container spacing={5}>
        <Grid item xs={6}>
          <UploadVCF onUploadComplete={handleFileUploadComplete}/>
        </Grid>
        <Grid item xs={6}>
        <Box width={330} mt={1} ml={25}>
            <img src="/static/new_images/dna-removebg-preview.png"/>
          </Box>
        </Grid>

        <Grid item xs={12}>
          <FilesTable isFileUploaded={isFileUploaded} resetFileUploaded={resetFileUploaded} />
        </Grid>
      </Grid>
    </Page>
  )
}

export default UploadView
