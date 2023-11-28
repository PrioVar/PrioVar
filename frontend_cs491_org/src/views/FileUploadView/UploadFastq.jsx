import { Box, Card, CardContent, CardHeader, LinearProgress, Typography, Stack } from '@material-ui/core'
import React, { useEffect, useState } from 'react'
import axios from 'axios'
import { uploadFile } from '../../api/fastq'
import { useImmer } from 'use-immer'
import UploadMultiPairFile from '../../components/upload/UploadMultiPairFile'
import { useFiles } from '../../api/file'

let cancelTokenSource = null

const toFileId = (file) => {
  return file.name
    .replace(/_R1/g, '')
    .replace(/_R2/g, '')
    .replace(/_I1/g, '')
    .replace(/_I2/g, '')
    .replace(/_1.fastq.gz$/, '')
    .replace(/_2.fastq.gz$/, '')
    .replace(/.fastq.gz$/, '')
    .replace(/.gz$/, '')
    .replace(/.fastq$/, '')
}

const UploadFastq = function () {
  const [fileQueue, setFileQueue] = useImmer(new Map())
  const [state, setState] = useState('IDLE')
  const [uploadProgress, setUploadProgress] = useState({ percentCompleted: 0, timeRemaining: 0 })
  const [uploadedFiles, setUploadedFiles] = useState([])
  const filesApi = useFiles()

  const getFilesToUpload = () => {
    const result = []
    fileQueue.forEach((entry) => {
      ;[entry.left, entry.right].forEach((file) => {
        if (file && !uploadedFiles.includes(file)) {
          result.push(file)
        }
      })
    })
    return result
  }

  const filesToUpload = getFilesToUpload()

  const handleDropMultiFile = async (filesToUpload) => {
    filesToUpload.forEach((file) => {
      const fileId = toFileId(file)

      const isRight = file.name.includes('_R2') || file.name.includes('_I2') || file.name.includes('_2.fastq.gz')

      setFileQueue((draft) => {
        if (!draft.has(fileId)) {
          draft.set(fileId, { left: null, right: null })
        }

        const entry = draft.get(fileId)
        if (isRight || entry.left) {
          entry.right = file
        } else {
          entry.left = file
        }
      })
    })

    setState('READY_TO_UPLOAD')
  }

  const handleUploadFiles = async (_filesToUpload) => {
    setState('UPLOADING')
  }

  const handleRemoveAll = () => {
    setFileQueue(() => new Map())
    setUploadProgress({ percentCompleted: 0, timeRemaining: 0 })
    setUploadedFiles([])
  }

  const handleRemove = (file) => {
    setFileQueue((draft) => {
      const fileId = toFileId(file)
      const entry = draft.get(fileId)

      if (entry.left === file) {
        entry.left = null
      } else if (entry.right === file) {
        entry.right = null
      }

      if (!entry.left && !entry.right) {
        draft.delete(fileId)
      }
    })
  }

  const handleCancel = () => {
    handleRemoveAll()
    cancelTokenSource.cancel('Cancelled by user')
  }

  useEffect(() => {
    const uploadFilesStateMachine = async () => {
      if (state === 'UPLOADING') {
        if (filesToUpload.length > 0) {
          const nextFile = filesToUpload[0]

          cancelTokenSource = axios.CancelToken.source()
          await uploadFile(nextFile, cancelTokenSource.token, ({ percentCompleted, timeRemaining }) => {
            setUploadProgress({ percentCompleted, timeRemaining })
          })
          setUploadProgress({ percentCompleted: 0, timeRemaining: 0 })
          setUploadedFiles([...uploadedFiles, nextFile])

          // handleRemove(nextFile)

          await filesApi.refresh()
        } else {
          // Done uploading all the files
          setState('IDLE')
          handleRemoveAll()
        }
      }
    }

    uploadFilesStateMachine()
  }, [state, fileQueue, uploadedFiles])

  return (
    <Card sx={{ mb: 3 }}>
      <CardHeader title="Upload FASTQ File" />
      <CardContent>
        <UploadMultiPairFile
          maxSize={1024 * 1024 * 1024 * 10} // 10 Gb
          files={[...fileQueue.values()]}
          onDrop={handleDropMultiFile}
          onRemove={handleRemove}
          onRemoveAll={handleRemoveAll}
          onUploadFiles={handleUploadFiles}
          locked={state === 'UPLOADING'}
          onCancel={handleCancel}
          uploadProgress={uploadProgress}
          uploadedFiles={uploadedFiles}
          activeFile={filesToUpload.length > 0 && state === 'UPLOADING' ? filesToUpload[0] : null}
        />
      </CardContent>
    </Card>
  )
}

export default UploadFastq
