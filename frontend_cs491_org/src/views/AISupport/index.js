import React from 'react'
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
