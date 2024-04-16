import React from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { Grid, FormControlLabel, Radio, RadioGroup } from '@material-ui/core'

import { setScenario } from 'src/redux/slices/variantFilters'

const SCENARIO_OPTIONS = [
  { label: 'none', value: 'NONE' },
  { label: 'dominant', value: 'DOMINANT' },
  { label: 'recessive', value: 'RECESSIVE' },
  { label: 'de novo', value: 'DE_NOVO' },
  { label: 'compound het', value: 'COMPOUND_HET' },
  { label: 'x linked', value: 'X_LINKED' },
]

function ScenarioFilters() {
  const scenarioFilter = useSelector((state) => state.scenarioFilter)
  const dispatch = useDispatch()

  const handleChange = (event) => {
    dispatch(setScenario(event.target.value))
  }

  return (
    <RadioGroup value={scenarioFilter} onChange={handleChange}>
      <Grid container>
        <Grid item xs={6} display="flex" flexDirection="column">
          {SCENARIO_OPTIONS.slice(0, 3).map(({ label, value }) => (
            <FormControlLabel key={value} control={<Radio />} value={value} label={label} />
          ))}
        </Grid>
        <Grid item xs={6} display="flex" flexDirection="column">
          {SCENARIO_OPTIONS.slice(3).map(({ label, value }) => (
            <FormControlLabel key={value} control={<Radio />} value={value} label={label} />
          ))}
        </Grid>
      </Grid>
    </RadioGroup>
  )
}

export default ScenarioFilters
