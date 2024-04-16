import React from 'react'
import PropTypes from 'prop-types'
import { Box, Accordion, AccordionDetails, AccordionSummary, Typography } from '@material-ui/core'

import { Icon } from '@iconify/react'
import arrowIosDownwardFill from '@iconify-icons/eva/arrow-ios-downward-fill'

function FilterAccordion({ onChange, expanded, title, children }) {
  const handleChange = (_e, expanded) => {
    onChange(expanded)
  }

  return (
    <Accordion expanded={expanded} onChange={handleChange} disableGutters>
      <AccordionSummary expandIcon={<Icon icon={arrowIosDownwardFill} width={20} height={20} />}>
        <Typography variant="subtitle1" pl={1}>
          {title}
        </Typography>
      </AccordionSummary>
      <AccordionDetails>
        <Box p={2}>{children}</Box>
      </AccordionDetails>
    </Accordion>
  )
}
FilterAccordion.propTypes = {
  onChange: PropTypes.func.isRequired,
  expanded: PropTypes.bool.isRequired,
  title: PropTypes.string.isRequired,
  children: PropTypes.node.isRequired,
}

export default FilterAccordion
