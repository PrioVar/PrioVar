import React from 'react'
import { useParams } from 'react-router-dom'
import { Box, Typography, Divider } from '@material-ui/core'
import Tags from 'src/components/Tags'
import { HPO_OPTIONS } from 'src/constants'
import { useHpo } from '../../api/vcf'

const HpoPanel = function () {
  const { fileId, sampleName } = useParams()
  const [hpoList, setHpoList] = useHpo({
    fileId,
  })

  return (
    <>
      <Box py={2} pr={1} pl={2.5}>
        <Typography variant="subtitle1" gutterBottom>
          HPO values for {sampleName}
        </Typography>
      </Box>
      <Divider />
      <Box p={2.5}>
        <Tags title={sampleName} options={HPO_OPTIONS} value={hpoList} onChange={setHpoList} />
      </Box>
    </>
  )
}

export default HpoPanel
