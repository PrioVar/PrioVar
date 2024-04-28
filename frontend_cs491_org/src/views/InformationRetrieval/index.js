import React from 'react'
//import Page from 'src/components/Page'
//import { Container, Grid } from '@material-ui/core'
//import ReactDOM from 'react-dom';
import { CssBaseline } from '@mui/material';
import InformationRetrievalChat from './InformationRetrievalChat'


// TODO: UploadVCF and UploadFastq should be abstracted into a single component
// Likewise for FilesTable

const InformationRetrieval = function () {
  return (
    <>
      <CssBaseline /> {/* Normalize the default browser styles */}
      <InformationRetrievalChat />
    </>
  )
}

export default InformationRetrieval
