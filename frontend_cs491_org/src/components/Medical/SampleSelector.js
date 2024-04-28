import { Box, LinearProgress, ToggleButton, ToggleButtonGroup } from '@material-ui/core'
import PropTypes from 'prop-types'
import React, { useEffect } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { useSampleName } from '../../api/vcf'

const SampleSelector = function ({ fileId = null, disabled = false }) {
  const params = useParams()
  // TODO: Shitcode
  if (!fileId) {
    fileId = params.fileId
  }

  const { status, data: sampleName = null } = useSampleName({ fileId })
  const navigate = useNavigate()

  const createHandleClick = (sampleName) => () => {
    navigate(`/priovar/variants/${fileId}/${sampleName}`)
  }

  switch (status) {
    case 'success':
      return (
        <ToggleButtonGroup value={sampleName} exclusive>
          <ToggleButton
            key={`${fileId}.${sampleName}`}
            value={sampleName}
            onClick={createHandleClick(sampleName)}
            disabled={disabled}
          >
            {sampleName}
          </ToggleButton>
        </ToggleButtonGroup>
      )
    default:
      return (
        <Box py={2}>
          <LinearProgress />
        </Box>
      )
  }
}

SampleSelector.propTypes = {
  fileId: PropTypes.string,
  disabled: PropTypes.bool,
}

export default SampleSelector
