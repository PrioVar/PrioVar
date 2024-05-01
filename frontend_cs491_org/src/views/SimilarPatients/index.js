import React from 'react'
import Page from 'src/components/Page'
import { Container, Grid } from '@material-ui/core'

import SimilarPatientsTable from './SimilarPatientsTable'


// TODO: UploadVCF and UploadFastq should be abstracted into a single component
// Likewise for FilesTable

const SimilarPatients = function () {
  return (
    <Page title="Searching Similar Patients" style={{
        top: 0, 
        left: 0, 
        width: '100%', 
        height: '100vh', 
        backgroundImage: 'url("/static/new_images/cells_and_stetes.png")', 
        backgroundSize: 'cover', 
        backgroundPosition: 'center center',
        backgroundColor: 'rgba(255, 255, 255, 0.8)', // Adds white transparency
        backgroundBlendMode: 'overlay' // This blends the background color with the image
      }}>
      <Grid container spacing={5}>

        <Grid item xs={12}>
          <SimilarPatientsTable />
        </Grid>

      </Grid>
    </Page>
  )
}

export default SimilarPatients
