import arrowIosDownwardFill from '@iconify-icons/eva/arrow-ios-downward-fill'

import { Icon } from '@iconify/react'
import { Accordion, AccordionDetails, AccordionSummary, Box, Stack, Typography } from '@material-ui/core'
import PropTypes from 'prop-types'
import React, { useRef, useState } from 'react'

const Section = function ({ title, children, action, height, defaultExpand }) {
  const [isExpanded, setIsExpanded] = useState(defaultExpand)
  const actionRef = useRef()

  const handleChange = (e, expanded) => {
    const allDescendants = [...(actionRef.current?.querySelectorAll('*') ?? [])]
    if (allDescendants.includes(e.target)) {
      return
    }

    setIsExpanded(expanded)
  }

  return (
    <Accordion expanded={isExpanded} onChange={handleChange} disableGutters sx={{ m: 1, py: 2 }}>
      <AccordionSummary expandIcon={<Icon icon={arrowIosDownwardFill} width={20} height={20} />} sx={{ mx: 1 }}>
        <Stack direction="row" alignItems="baseline" spacing={1.5}>
          <Typography variant="h5">{title}</Typography>
          <Box ref={actionRef}>{action}</Box>
        </Stack>
      </AccordionSummary>
      <AccordionDetails>
        <Box p={2} height={height}>
          {children}
        </Box>
      </AccordionDetails>
    </Accordion>
  )
}

Section.propTypes = {
  title: PropTypes.string.isRequired,
  children: PropTypes.node.isRequired,
}

export default Section
