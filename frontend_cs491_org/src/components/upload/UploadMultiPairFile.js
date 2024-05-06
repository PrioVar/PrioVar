import { isString } from 'lodash'
import PropTypes from 'prop-types'
import { Icon } from '@iconify/react'
import { useDropzone } from 'react-dropzone'
import CloudDoneIcon from '@material-ui/icons/CloudDone'
import CloudUploadIcon from '@material-ui/icons/CloudUpload'
import CloudIcon from '@material-ui/icons/Cloud'
import QuestionMarkIcon from '@iconify-icons/eva/question-mark-outline'
import closeFill from '@iconify/icons-eva/close-fill'
import { motion, AnimatePresence } from 'framer-motion'
// material
import { alpha, styled } from '@material-ui/core/styles'
import {
  Box,
  List,
  Stack,
  Paper,
  Button,
  ListItem,
  Typography,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  Grid,
  LinearProgress,
} from '@material-ui/core'
// utils
import { fData } from '../../utils/formatNumber'
//
import { MIconButton } from '../@material-extend'
import { varFadeInRight } from '../animate'
import { UploadIllustration } from '../../assets'
import { fDuration } from '../../utils/formatTime'

// ----------------------------------------------------------------------

const DropZoneStyle = styled('div')(({ theme }) => ({
  outline: 'none',
  display: 'flex',
  textAlign: 'center',
  alignItems: 'center',
  flexDirection: 'column',
  justifyContent: 'center',
  padding: theme.spacing(5, 1),
  borderRadius: theme.shape.borderRadius,
  backgroundColor: theme.palette.background.neutral,
  border: `1px dashed ${theme.palette.grey[500_32]}`,
  '&:hover': { opacity: 0.72, cursor: 'pointer' },
  [theme.breakpoints.up('md')]: { textAlign: 'left', flexDirection: 'row' },
}))

// ----------------------------------------------------------------------

UploadMultiPairFile.propTypes = {
  error: PropTypes.bool,
  files: PropTypes.array,
  onRemove: PropTypes.func,
  onRemoveAll: PropTypes.func,
  sx: PropTypes.object,
  onUploadFiles: PropTypes.func,
  locked: PropTypes.bool,
}

export default function UploadMultiPairFile({
  error,
  files,
  onRemove,
  onRemoveAll,
  onUploadFiles,
  locked = false,
  sx,
  onCancel,
  uploadProgress,
  activeFile,
  uploadedFiles,
  ...other
}) {
  const hasFile = files.length > 0

  const { getRootProps, getInputProps, isDragActive, isDragReject, fileRejections } = useDropzone({
    ...other,
  })

  const ShowRejectionItems = function () {
    return (
      <Paper
        variant="outlined"
        sx={{
          py: 1,
          px: 2,
          mt: 3,
          borderColor: 'error.light',
          bgcolor: (theme) => alpha(theme.palette.error.main, 0.08),
        }}
      >
        {fileRejections.map(({ file, errors }) => {
          const { path, size } = file
          return (
            <Box key={path} sx={{ my: 1 }}>
              <Typography variant="subtitle2" noWrap>
                {path} - {fData(size)}
              </Typography>
              {errors.map((e) => (
                <Typography key={e.code} variant="caption" component="p">
                  - {e.message}
                </Typography>
              ))}
            </Box>
          )
        })}
      </Paper>
    )
  }

  return (
    <Box sx={{ width: '100%', ...sx }}>
      <DropZoneStyle
        {...getRootProps()}
        sx={{
          ...(isDragActive && { opacity: 0.72 }),
          ...((isDragReject || error) && {
            color: 'error.main',
            borderColor: 'error.light',
            bgcolor: 'error.lighter',
          }),
        }}
      >
        <input {...getInputProps()} />

        <UploadIllustration sx={{ width: 220 }} />

        <Box sx={{ p: 3, ml: { md: 2 } }}>
          <Typography gutterBottom variant="h5">
            Drop or Select file
          </Typography>

          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
            Drop files here or click&nbsp;
            <Typography variant="body2" component="span" sx={{ color: 'primary.main', textDecoration: 'underline' }}>
              browse
            </Typography>
            &nbsp;through your machine
          </Typography>
        </Box>
      </DropZoneStyle>

      {fileRejections.length > 0 && <ShowRejectionItems />}

      <List disablePadding sx={{ ...(hasFile && { my: 3 }) }}>
        <AnimatePresence>
          {files.map(({ left, right }) => {
            return (
              <Grid container spacing={1}>
                {[left, right].map((file) => {
                  const { name = 'Missing FASTQ file...', size = 0 } = file || {}
                  const key = isString(file) ? file : name

                  const isUploaded = uploadedFiles.includes(file)
                  const isUploading = activeFile && activeFile === file
                  const isMissing = !file
                  const isPending = !isUploaded && !isUploading && !isMissing

                  const getSecondaryText = () => {
                    if (!file || isString(file)) {
                      return ''
                    }
                    if (isUploading) {
                      return [
                        fData(size),
                        `${uploadProgress.percentCompleted}%`,
                        `${fDuration(uploadProgress.timeRemaining)} left`,
                      ].join(' • ')
                    }
                    return fData(size)
                  }

                  return (
                    <Grid item xs={6}>
                      <ListItem
                        key={key}
                        component={motion.div}
                        {...varFadeInRight}
                        alignItems="flex-start"
                        sx={{
                          my: 1,
                          py: 0.75,
                          px: 2,
                          borderRadius: 1,
                          border: (theme) => `solid 1px ${theme.palette.divider}`,
                          bgcolor: file ? 'background.paper' : 'action.disabledBackground',
                          flexDirection: 'column',
                        }}
                      >
                        <Stack direction="row">
                          <ListItemIcon>
                            {isMissing && <Icon icon={QuestionMarkIcon} width={28} height={28} />}
                            {isUploading && <CloudUploadIcon />}
                            {isUploaded && <CloudDoneIcon />}
                            {isPending && <CloudIcon />}
                          </ListItemIcon>
                          <ListItemText
                            primary={isString(file) ? file : name}
                            secondary={getSecondaryText()}
                            primaryTypographyProps={{
                              variant: 'subtitle2',
                              color: file ? undefined : 'action.disabled',
                            }}
                            secondaryTypographyProps={{ variant: 'caption' }}
                          />
                          <ListItemSecondaryAction>
                            {!locked && file && (
                              <MIconButton edge="end" size="small" onClick={() => onRemove(file)}>
                                <Icon icon={closeFill} />
                              </MIconButton>
                            )}
                          </ListItemSecondaryAction>
                        </Stack>
                        {isUploading && (
                          <LinearProgress
                            variant="determinate"
                            value={uploadProgress.percentCompleted}
                            sx={{ width: '100%', mt: 0.5 }}
                          />
                        )}
                      </ListItem>
                    </Grid>
                  )
                })}
              </Grid>
            )
          })}
        </AnimatePresence>
      </List>

      {hasFile && (
        <Stack direction="row" justifyContent="flex-end">
          <Button onClick={onCancel} sx={{ mr: 1.5 }} disabled={!locked}>
            Cancel
          </Button>
          <Button onClick={onRemoveAll} sx={{ mr: 1.5 }} disabled={locked}>
            Remove 
          </Button>
          <Button variant="contained" onClick={() => onUploadFiles(files)} disabled={locked}>
            Upload file
          </Button>
        </Stack>
      )}
    </Box>
  )
}
