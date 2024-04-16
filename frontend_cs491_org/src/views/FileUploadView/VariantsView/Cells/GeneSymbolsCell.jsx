import { Box, Typography } from '@material-ui/core'
import PropTypes from 'prop-types'
import React from 'react'
import ExpandOnClick from 'src/components/ExpandOnClick'
import ResponsiveGrid from 'src/components/ResponsiveGrid'
import { ChipLink } from 'src/views/VariantsView/Cells/ChipLink'

const GeneSymbolsCell = function ({ symbols, limit = 1 }) {
  const renderedChipLinks = symbols.map((symbol) => (
    <ChipLink
      key={symbol}
      href={`https://www.genecards.org/cgi-bin/carddisp.pl?gene=${encodeURIComponent(symbol)}`}
      label={symbol}
      variant="filled"
    />
  ))

  return (
    <ExpandOnClick
      expanded={
        <ResponsiveGrid columns={3} rowSpacing={0.5} columnSpacing={0.5} p={3}>
          {renderedChipLinks}
        </ResponsiveGrid>
      }
    >
      {({ ref, onClick }) => (
        <Box ref={ref} onClick={renderedChipLinks.length > 1 ? onClick : () => {}} p={1}>
          <ResponsiveGrid columns={3} rowSpacing={0.5} columnSpacing={0.5}>
            {renderedChipLinks.slice(0, limit)}
          </ResponsiveGrid>
        </Box>
      )}
    </ExpandOnClick>
  )
}

GeneSymbolsCell.propTypes = {
  symbols: PropTypes.arrayOf(PropTypes.string).isRequired,
}

export default GeneSymbolsCell
