import React from 'react'
import Page from 'src/components/Page'
import { Container, Grid } from '@material-ui/core'
import ReactDOM from 'react-dom';
import { CssBaseline } from '@mui/material';
import AISupportChat from './AISupportChat'


// TODO: UploadVCF and UploadFastq should be abstracted into a single component
// Likewise for FilesTable

const AISupport = function () {
  return (
    <>
      <CssBaseline /> {/* Normalize the default browser styles */}
      <AISupportChat />
    </>
  )
}

export default AISupport
