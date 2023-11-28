import { Box, Stack } from '@material-ui/core'
import { styled } from '@material-ui/core/styles'
import PropTypes from 'prop-types'
import React from 'react'
import GroupedItems from 'src/components/GroupedItems'
import ResponsiveGrid from 'src/components/ResponsiveGrid'
import { isValid } from 'src/utils/validation'
import { LabelledFloat } from 'src/views/VariantsView/Cells/LabelledFloat'

const valueToColor = (value) => {
  if (value <= 0.0001) {
    return 'error'
  }
  if (value <= 0.1) {
    return 'warning'
  }
  return 'success'
}

const FrequencyCell = function ({ libraAf, gnomAdAf, oneKgAf, tvAf, exAcAf }) {
  return (
    <GroupedItems
      component={ResponsiveGrid}
      columns={3}
      rowSpacing={0.5}
      columnSpacing={1}
      alignItems="center"
      justifyContent="center"
      p={1}
    >
      {isValid(libraAf) && <LabelledFloat label="Local" value={libraAf} color={valueToColor(libraAf)} />}
      {isValid(gnomAdAf) && <LabelledFloat label="gnomAD" value={gnomAdAf} color={valueToColor(gnomAdAf)} />}
      {isValid(oneKgAf) && <LabelledFloat label="1KG" value={oneKgAf} color={valueToColor(oneKgAf)} />}
      {isValid(exAcAf) && <LabelledFloat label="ExAC" value={exAcAf} color={valueToColor(exAcAf)} />}
      {isValid(tvAf) && <LabelledFloat label="TurkishVariome" value={tvAf} color={valueToColor(tvAf)} />}
    </GroupedItems>
  )
}

FrequencyCell.propTypes = {
  libraAf: PropTypes.number,
  gnomAdAf: PropTypes.number,
  oneKgAf: PropTypes.number,
  tvAf: PropTypes.number,
  exAcAf: PropTypes.number,
}

export default FrequencyCell
