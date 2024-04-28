import { Box, Card, CardContent, CardHeader, LinearProgress, Typography, Stack } from '@material-ui/core'
import React, { useEffect, useState } from 'react'
import { UploadMultiFile } from 'src/components/upload'
import axios from 'axios'
import { uploadFile } from '../../api/vcf'
import { useFiles } from '../../api/file/list'

let cancelTokenSource = null

const UploadVCF = function ({ onUploadComplete }) {
  const [fileQueue, setFileQueue] = useState([])
  const [state, setState] = useState('IDLE')
  const [uploadProgress, setUploadProgress] = useState(0)
  const filesApi = useFiles()

  const handleDropMultiFile = async (filesToUpload) => {
    setFileQueue(filesToUpload)
    setState('READY_TO_UPLOAD')

    // await uploadFile(fileToUpload)
    // await filesApi.refresh()
  }

  const handleUploadFiles = async (_filesToUpload) => {
    setState('UPLOADING')
  }

  const handleRemoveAll = () => {
    setFileQueue([])
  }

  const handleRemove = (file) => {
    setFileQueue(fileQueue.filter((f) => f !== file))
  }

  const handleCancel = () => {
    setFileQueue([])
    cancelTokenSource.cancel('Cancelled by user')
  }

  useEffect(() => {
    const uploadFilesStateMachine = async () => {
      if (state === 'UPLOADING') {
        if (fileQueue.length > 0) {
          setUploadProgress(0)
          cancelTokenSource = axios.CancelToken.source()
          await uploadFile(fileQueue[0], (value) => setUploadProgress(value), cancelTokenSource.token)

          setFileQueue(fileQueue.slice(1))
          await filesApi.refresh()
        } else {
          // Done uploading all the files
          setState('IDLE')
          onUploadComplete(true);
        }
      }
    }

    uploadFilesStateMachine()
  }, [state, fileQueue, onUploadComplete])

  return (
    <Card sx={{ mt: 1 }}>
      <CardHeader title="Upload VCF File" />
      <CardContent>
        {state === 'UPLOADING' && fileQueue.length > 0 && (
          <Box py={2} px={1}>
            <Box display="flex" flexDirection="row">
              <Typography variant="subtitle1" color="textSecondary">
                Uploading&nbsp;
              </Typography>
              <Typography variant="subtitle1" color="textPrimary">
                {fileQueue[0].name}
              </Typography>
              <Typography variant="subtitle1" color="textSecondary" pl={1}>
                ({uploadProgress}%)&nbsp;...
              </Typography>
            </Box>
            <Box p={1} />
            <LinearProgress variant="determinate" value={uploadProgress} />
          </Box>
        )}
        <UploadMultiFile
          maxSize={1024 * 1024 * 1024}
          files={fileQueue}
          onDrop={handleDropMultiFile}
          onRemove={handleRemove}
          onRemoveAll={handleRemoveAll}
          onUploadFiles={handleUploadFiles}
          locked={state === 'UPLOADING'}
          onCancel={handleCancel}
        />
      </CardContent>
    </Card>
  )
}

export default UploadVCF
