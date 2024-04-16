import React from 'react'
import PropTypes from 'prop-types'
import { Accordion, AccordionDetails, AccordionSummary, Box, Typography } from '@material-ui/core'
import InfoIcon from '@mui/icons-material/Info'
import IconButton from '@mui/material/IconButton'
import ToolTip from '@mui/material/Tooltip'

import { Icon } from '@iconify/react'
import arrowIosDownwardFill from '@iconify-icons/eva/arrow-ios-downward-fill'

function FiltersAccordion({ title, children, tooltip }) {
  return (
    <Accordion disableGutters color="green">
      <AccordionSummary expandIcon={<Icon icon={arrowIosDownwardFill} width={20} height={20} />}>
        <Typography variant="subtitle1" pl={1}>
          {title}
          {tooltip && (
            <ToolTip title={tooltip}>
              <IconButton>
                <InfoIcon />
              </IconButton>
            </ToolTip>
          )}
        </Typography>
      </AccordionSummary>
      <AccordionDetails>
        <Box p={2}>{children}</Box>
      </AccordionDetails>
    </Accordion>
  )
}
FiltersAccordion.propTypes = {
  title: PropTypes.string.isRequired,
  children: PropTypes.node.isRequired,
}

export default FiltersAccordion
