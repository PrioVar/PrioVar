import {
//  Box,
  Divider,
//  FilledInput,
  Grid,
  MenuItem,
  Select,
  Stack,
//  TextField,
//  ToggleButton,
//  ToggleButtonGroup,
//  Tooltip,
//  Typography,
} from '@material-ui/core'
//import { countBy } from 'lodash'
//import { maxBy, reduce } from 'ramda'
import React, { useState } from 'react'
import { GENES_TAB_HEIGHT } from 'src/constants'
import DiseaseBarChart from 'src/views/VariantDetailsView/GenesTab/DiseasesBarChart'
import OmimTable from 'src/views/VariantDetailsView/GenesTab/OmimTable'
import CustomPhenotypes from './CustomPhenotypes'
import Section from '../Section'


const GenesTab = function ({ disease, transcripts, omim, variantId, activeGene }) {
  const activeDisease = disease[activeGene]
  const activeOmim = omim[activeGene]

  // const impactToColorMap = {
  //   LOW: 'success',
  //   MODERATE: 'warning',
  //   HIGH: 'error',
  //   MODIFIER: 'primary',
  // }
  //
  // const geneImpactMap = calculateGeneImpactMap(transcripts)

  return (
    <Stack direction="column">
      {/*<Stack direction="row" alignItems="center">*/}
      {/*  /!* <Typography variant="body2">Selected gene:</Typography> *!/*/}
      {/*  /!* <Box p={0.5} /> *!/*/}
      {/*  <ToggleButtonGroup value={activeGene} exclusive>*/}
      {/*    {Object.keys(disease)*/}
      {/*      .sort()*/}
      {/*      .filter((geneId) => geneId && geneId !== 'null')*/}
      {/*      .map((geneId) => {*/}
      {/*        const impacts = geneImpactMap[geneId]*/}
      {/*        const maxImpact = getMaxImpact(impacts)*/}
      {/*        const color = impactToColorMap[maxImpact]*/}

      {/*        // See: https://github.com/mui-org/material-ui/issues/12921#issuecomment-422495422*/}
      {/*        return (*/}
      {/*          <ToggleButtonWithTooltip*/}
      {/*            key={geneId}*/}
      {/*            value={geneId}*/}
      {/*            onClick={() => setActiveGene(geneId)}*/}
      {/*            color={color}*/}
      {/*            tooltipProps={{*/}
      {/*              title: getTooltipTitle(impacts),*/}
      {/*              arrow: true,*/}
      {/*              placement: 'top',*/}
      {/*            }}*/}
      {/*          >*/}
      {/*            {geneId}*/}
      {/*          </ToggleButtonWithTooltip>*/}
      {/*        )*/}
      {/*      })}*/}
      {/*  </ToggleButtonGroup>*/}
      {/*</Stack>*/}
      {/* <Box p={1} />
      <Typography variant="subtitle1">Related Diseases</Typography> */}
      <CustomPhenotypes geneId={activeGene} />
      <Divider sx={{ my: 2 }} />
      <Grid container spacing={1}>
        {activeDisease && (
          <Grid item xs={7} sx={{ height: GENES_TAB_HEIGHT }}>
            <DiseaseBarChart disease={activeDisease} activeGene={activeGene} />
          </Grid>
        )}
        <Grid item xs={5}>
          <OmimTable omim={activeOmim} geneSymbol={activeGene} />
        </Grid>
      </Grid>
    </Stack>
  )
}

function GenesTabSection(props) {
  const { disease } = props
  const geneSymbols = Object.keys(disease).sort()
  const [activeGene, setActiveGene] = useState(geneSymbols[0])

  return (
    <Section
      title="Genes"
      action={
        <Select
          value={activeGene}
          onChange={(e) => setActiveGene(e.target.value)}
          autoWidth
          MenuProps={{ disablePortal: true }}
          variant="filled"
          size="small"
          inputProps={{
            sx: { py: 0.5 },
          }}
        >
          {geneSymbols.map((geneSymbol) => (
            <MenuItem key={geneSymbol} value={geneSymbol}>
              {geneSymbol}
            </MenuItem>
          ))}
        </Select>
      }
    >
      <GenesTab {...props} activeGene={activeGene} />
    </Section>
  )
}

export default GenesTabSection

/*
const maxByImpact = maxBy((impact) => ['MODIFIER', 'LOW', 'MODERATE', 'HIGH'].indexOf(impact))
const getMaxImpact = reduce(maxByImpact, 'MODIFIER')

const getTooltipTitle = (labels) => {
  const countMap = countBy(labels)

  return Object.entries(countMap)
    .map(([label, count]) => `${count} ✖ ${label}`)
    .join(', ')
}

const ToggleButtonWithTooltip = function ({ tooltipProps, ...props }) {
  return (
    <Tooltip {...tooltipProps}>
      <ToggleButton {...props} />
    </Tooltip>
  )
}

const calculateGeneImpactMap = (transcripts) => {
  const result = {}

  transcripts.forEach(({ Gene_Symbol: geneSymbol, Impact: impact }) => {
    if (!result[geneSymbol]) {
      result[geneSymbol] = []
    }
    result[geneSymbol].push(impact)
  })

  return result
}
*/