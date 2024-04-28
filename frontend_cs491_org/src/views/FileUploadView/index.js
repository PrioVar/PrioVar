import React from 'react'
import { useState } from 'react'
import Page from 'src/components/Page'
import { Grid, Box } from '@material-ui/core'

import FilesTable from './FilesTable'
import UploadVCF from './UploadVCF'


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
    <Page title="Upload VCF | PrioVar" style={{
        top: 0, 
        left: 0, 
        width: '100%', 
        height: '100vh', 
        backgroundImage: 'url("/static/new_images/another_dna.png")', 
        backgroundSize: 'cover', 
        backgroundPosition: 'center center',
        backgroundColor: 'rgba(255, 255, 255, 0.5)', // Adds white transparency
        backgroundBlendMode: 'overlay' // This blends the background color with the image
      }}>
      <Grid container spacing={5} >
        <Grid item xs={6}>
          <UploadVCF onUploadComplete={handleFileUploadComplete}/>
        </Grid>
        <Grid item xs={6}>

        </Grid>

        <Grid item xs={12}>
          <FilesTable isFileUploaded={isFileUploaded} resetFileUploaded={resetFileUploaded} />
        </Grid>
      </Grid>
    </Page>
  )
}

export default UploadView
