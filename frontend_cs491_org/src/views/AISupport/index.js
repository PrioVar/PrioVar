import React from 'react'
import { CssBaseline } from '@mui/material';
import AISupportChat from './AISupportChat'
import Page from 'src/components/Page'


// TODO: UploadVCF and UploadFastq should be abstracted into a single component
// Likewise for FilesTable

const AISupport = function () {
  return (
    <Page title="AI Support" style={{
        top: 0, 
        left: 0, 
        width: '100%', 
        height: '100vh', 
        backgroundImage: 'url("/static/new_images/ai_dna_1.png")', 
        backgroundSize: 'cover', 
        backgroundPosition: 'center center',
        backgroundColor: 'rgba(255, 255, 255, 0.7)', // Adds white transparency
        backgroundBlendMode: 'overlay' // This blends the background color with the image
      }}>
      <CssBaseline /> {/* Normalize the default browser styles */}
      <AISupportChat />
    </Page>
  )
}

export default AISupport
