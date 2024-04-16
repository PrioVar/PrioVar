import React from 'react'
import { Box, FormGroup, Grid } from '@material-ui/core'
import { IMPACT_CHOICES, CLIN_VAR_SEVERITY } from 'src/constants'
import { ACMG_CHOICES } from 'src/constants/acmg'
import ImpactChoiceChip from './ImpactChoiceChip'
import Label from 'src/components/Label'
import { useState, useEffect } from 'react'

function ImpactChoices(props) {
  const [impactList, setImpactList] = useState([])

  const addToImpactList = (value) => {
    setImpactList((impactList) => [...impactList, value])
  }

  const removeFromImpactList = (value) => {
    setImpactList((list) => list.filter((item) => item !== value))
  }

  const handleImpactArray = (value) => {
    if (impactList.includes(value)) removeFromImpactList(value)
    else addToImpactList(value)
  }

  useEffect(() => {
    props.setImpactEvent(impactList)
  }, [impactList, props])

  const selectAll = (title) => {
    const clickedValues =
      props.filterType === 'impact'
        ? IMPACT_CHOICES[title].map((i) => i.value)
        : CLIN_VAR_SEVERITY[title].map((i) => i.value)
    clickedValues.forEach(handleImpactArray)
  }

  const renderImpactChoiceGroup = ({ color, title, items }) => {
    return (
      <>
        {props.filterType !== 'acmg' && (
          <Label sx={{ pointer: 'cursor' }} color={color} onClick={() => selectAll(title)}>
            {title}
          </Label>
        )}
        <FormGroup>
          <Box p={2}>
            <Grid container spacing={1}>
              {items.map(({ label, value }) => (
                <Grid item xs key={value}>
                  <ImpactChoiceChip
                    color={color}
                    key={value}
                    label={label}
                    onClick={() => handleImpactArray(value)}
                    isSelected={value}
                    impactList={impactList}
                  />
                </Grid>
              ))}
            </Grid>
          </Box>
        </FormGroup>
      </>
    )
  }

  return (
    (props.filterType === 'impact' && (
      <Box display="flex" flexDirection="column">
        {renderImpactChoiceGroup({ color: 'error', title: 'HIGH', items: IMPACT_CHOICES.HIGH })}
        {renderImpactChoiceGroup({ color: 'warning', title: 'MODERATE', items: IMPACT_CHOICES.MODERATE })}
        {renderImpactChoiceGroup({ color: 'success', title: 'LOW', items: IMPACT_CHOICES.LOW })}
        {renderImpactChoiceGroup({ color: 'default', title: 'MODIFIER', items: IMPACT_CHOICES.MODIFIER })}
      </Box>
    )) ||
    (props.filterType === 'acmg' && (
      <Box display="flex" flexDirection="column">
        {renderImpactChoiceGroup({
          color: 'default',
          title: 'ACMG',
          items: ACMG_CHOICES.DESCRIPTIONS,
        })}
      </Box>
    )) ||
    (props.filterType === 'clinvar' && (
      <Box display="flex" flexDirection="column">
        {renderImpactChoiceGroup({
          color: 'error',
          title: 'HIGH',
          items: CLIN_VAR_SEVERITY.HIGH,
        })}
        {renderImpactChoiceGroup({ color: 'default', title: 'MODERATE', items: CLIN_VAR_SEVERITY.MODERATE })}
        {renderImpactChoiceGroup({
          color: 'success',
          title: 'LOW',
          items: CLIN_VAR_SEVERITY.LOW,
        })}
      </Box>
    ))
  )
}

export default ImpactChoices
