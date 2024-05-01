import React from 'react'
import Page from 'src/components/Page'
import { Container, Grid } from '@material-ui/core'

import MyPatientsTable from './MyPatientsTable'


// TODO: UploadVCF and UploadFastq should be abstracted into a single component
// Likewise for FilesTable

const MyPatients = function () {
  return (
    <Page title="My Patients | PrioVar" >
      <Grid container spacing={5}>

        <Grid item xs={12} style={{
        top: 0, 
        left: 0, 
        width: '100%', 
        height: '100vh', 
        backgroundImage: 'url("/static/new_images/dna-helix.png")', 
        backgroundSize: 'cover', 
        backgroundPosition: 'center center',
        backgroundColor: 'rgba(255, 255, 255, 0.7)', // Adds white transparency
        backgroundBlendMode: 'overlay' // This blends the background color with the image
      }}>
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
