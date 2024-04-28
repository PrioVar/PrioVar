import React from 'react'
//import Page from 'src/components/Page'
//import { Container, Grid } from '@material-ui/core'
//import ReactDOM from 'react-dom';
import { CssBaseline } from '@mui/material';
import InformationRetrievalChat from './InformationRetrievalChat'
import Page from 'src/components/Page'



// TODO: UploadVCF and UploadFastq should be abstracted into a single component
// Likewise for FilesTable

const InformationRetrieval = function () {
  return (
    <Page title="AI Support" style={{
        top: 0, 
        left: 0, 
        width: '100%', 
        height: '100vh', 
        backgroundImage: 'url("/static/new_images/ai_dna_2.png")', 
        backgroundSize: 'cover', 
        backgroundPosition: 'center center',
        backgroundColor: 'rgba(255, 255, 255, 0.7)', // Adds white transparency
        backgroundBlendMode: 'overlay' // This blends the background color with the image
      }}>
      <CssBaseline /> {/* Normalize the default browser styles */}
      <InformationRetrievalChat />
    </Page>
  )
}

export default InformationRetrieval
