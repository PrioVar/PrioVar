import PropTypes from 'prop-types'
import React from 'react'
import ClickAwayTooltip from 'src/components/ClickAwayTooltip'
import Label from 'src/components/Label'
import { isHomogeneous } from 'src/utils/bio'

const GtCell = function ({ gt }) {
  return (
    <ClickAwayTooltip title={gt} arrow>
      <Label variant="ghost" sx={{ width: 40 }}>
        {isHomogeneous(gt) ? 'hom' : 'het'}
      </Label>
    </ClickAwayTooltip>
  )
}

GtCell.propTypes = {
  gt: PropTypes.string.isRequired,
}

export default GtCell
