import { Chip } from '@material-ui/core'
import DoneIcon from '@material-ui/icons/Done'
import PropTypes from 'prop-types'
import React from 'react'
import { useState, useEffect } from 'react'

function ImpactChoiceChip({ color, onClick, label, isSelected, impactList }) {
  const [selected, setSelected] = useState()

  const getFromImpactList = (value) => {
    let getValueCheck = false
    impactList.forEach((item) => {
      if (item === value) getValueCheck = true
    })
    setSelected(getValueCheck)
  }

  useEffect(() => {
    getFromImpactList(isSelected)
  }, [onClick])

  return (
    <Chip
      sx={{ width: '100%' }}
      color={selected ? color : 'default'}
      variant={selected ? 'filled' : 'outlined'}
      label={label}
      clickable
      onClick={onClick}
      onDelete={onClick}
      deleteIcon={<DoneIcon style={{ visibility: selected ? 'unset' : 'hidden' }} />}
    />
  )
}

ImpactChoiceChip.propTypes = {
  color: PropTypes.string.isRequired,
  onClick: PropTypes.func.isRequired,
  label: PropTypes.string.isRequired,
  isSelected: PropTypes.bool.isRequired,
}

export default ImpactChoiceChip
