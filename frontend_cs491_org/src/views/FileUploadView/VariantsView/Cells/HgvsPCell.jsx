import PropTypes from 'prop-types'
import React from 'react'
import HgvsCCell from 'src/views/VariantsView/Cells/HgvsCCell'

const HgvsPCell = function ({ value }) {
  return <HgvsCCell value={value} />
}

HgvsPCell.propTypes = {
  value: PropTypes.array.isRequired,
}

export default HgvsPCell
