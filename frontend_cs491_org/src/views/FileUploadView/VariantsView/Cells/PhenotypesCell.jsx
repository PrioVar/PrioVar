import { Typography } from '@material-ui/core'
import PropTypes from 'prop-types'
import React from 'react'
import { chain as flatMap } from 'ramda'
import ExpandOnClick from 'src/components/ExpandOnClick'

const MAX_ITEMS = 4

const PhenotypesCell = function ({ phenotypes }) {
  const multilineText = flatMap((text) => [text, <br />], phenotypes)

  return (
    <ExpandOnClick
      expanded={
        <Typography noWrap variant="body1" sx={{ p: 2 }}>
          {multilineText}
        </Typography>
      }
    >
      {({ ref, onClick }) => (
        <Typography noWrap variant="body2" letterSpacing={-0.25} ref={ref} onClick={onClick}>
          {multilineText.slice(0, MAX_ITEMS * 2)}
          {/*          {phenotypes.length > MAX_ITEMS && '…'} */}
        </Typography>
      )}
    </ExpandOnClick>
  )
}

PhenotypesCell.propTypes = {
  phenotypes: PropTypes.arrayOf(PropTypes.string).isRequired,
}

export default PhenotypesCell
