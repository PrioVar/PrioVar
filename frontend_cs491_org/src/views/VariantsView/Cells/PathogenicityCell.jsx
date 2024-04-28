import { Stack } from '@material-ui/core'
import PropTypes from 'prop-types'
import React from 'react'
import ResponsiveGrid from 'src/components/ResponsiveGrid'
import { isValid } from 'src/utils/validation'
import ClinVarCell from 'src/views/VariantsView/Cells/ClinVarCell'
import { LabelledFloat } from 'src/views/VariantsView/Cells/LabelledFloat'
import GroupedItems from 'src/components/GroupedItems'

// TODO: Refactor LabelledFloat and stuff into one common component

const valueToColor = (value) => {
  if (value >= 0.7) {
    return 'error'
  }
  if (value >= 0.5) {
    return 'warning'
  }
  return 'success'
}

const PathogenicityCell = function ({ clinVar, sift, polyphen, cadd, revel, libraP, dann, metalR }) {
  return (
    <GroupedItems component={Stack} direction="column" px={1} py={2}>
      {/* <LibraPathogenicityCell value={libraP} /> */}
      <ResponsiveGrid columns={3} rowSpacing={0.5} columnSpacing={1} alignItems="center" justifyContent="center">
        {isValid(clinVar) && <ClinVarCell label="ClinVar" value={clinVar} />}
        {isValid(cadd) && <LabelledFloat label="CADD" value={cadd} color={valueToColor(cadd)} />}
        {isValid(revel) && <LabelledFloat label="REVEL" value={revel} color={valueToColor(revel)} />}
        {isValid(dann) && <LabelledFloat label="DANN" value={dann} color={valueToColor(dann)} />}
        {isValid(metalR) && <LabelledFloat label="MetalR" value={metalR} color={valueToColor(metalR)} />}
        {isValid(polyphen) && <LabelledFloat label="Polyphen" value={polyphen} color={valueToColor(polyphen)} />}
        {isValid(sift) && <LabelledFloat label="SIFT" value={sift} color={valueToColor(sift)} />}
      </ResponsiveGrid>
    </GroupedItems>
  )
}

PathogenicityCell.propTypes = {
  sift: PropTypes.number,
  polyphen: PropTypes.number,
  cadd: PropTypes.number,
  revel: PropTypes.number,
  libraP: PropTypes.number,
  metalR: PropTypes.number,
  dann: PropTypes.number,
  clinVar: PropTypes.any,
}

export default PathogenicityCell
