import React from 'react'
import { Checkbox, Divider, Grid, Stack, Typography } from '@material-ui/core'
import YesNoAnyButtonGroup from '../YesNoAnyButtonGroup'
import MinMaxField from 'src/views/VariantsView/Toolbar/FiltersPanelMenu/MinMaxField'

function FilterBase({ enabled, onToggle, onChange, value, label }) {
  return (
    <Grid container direction="row" alignItems="center">
      <Grid item xs={1}>
        <Checkbox checked={enabled} onChange={onToggle} />
      </Grid>
      <Grid item xs={5}>
        <Typography variant="body1" color={enabled ? undefined : 'text.disabled'}>
          {label}
        </Typography>
      </Grid>
      <Grid item xs={6}>
        <MinMaxField title={label} value={value} onChange={onChange} disabled={!enabled} />
      </Grid>
    </Grid>
  )
}

function FrequencyFilters(props) {
  const handleAfToggle = () => {
    props.setToggleSelectorAf(!props.toggleSelectorAf)
    props.setIsToggleChangedFrequency([
      !props.isToggleChangedFrequency[0],
      props.isToggleChangedFrequency[1],
      props.isToggleChangedFrequency[2],
      props.isToggleChangedFrequency[3],
      props.isToggleChangedFrequency[4],
    ])
  }

  const handleOneKgToggle = () => {
    props.setToggleSelectorOneKgFrequency(!props.toggleSelectorOneKgFrequency)
    props.setIsToggleChangedFrequency([
      props.isToggleChangedFrequency[0],
      !props.isToggleChangedFrequency[1],
      props.isToggleChangedFrequency[2],
      props.isToggleChangedFrequency[3],
      props.isToggleChangedFrequency[4],
    ])
  }

  const handleGnomAdFrequencyToggle = () => {
    props.setToggleSelectorGnomAdFrequency(!props.toggleSelectorGnomAdFrequency)
    props.setIsToggleChangedFrequency([
      props.isToggleChangedFrequency[0],
      props.isToggleChangedFrequency[1],
      !props.isToggleChangedFrequency[2],
      props.isToggleChangedFrequency[3],
      props.isToggleChangedFrequency[4],
    ])
  }

  const handleExAcAfToggle = () => {
    props.setToggleSelectorExAcAf(!props.toggleSelectorExAcAf)
    props.setIsToggleChangedFrequency([
      props.isToggleChangedFrequency[0],
      props.isToggleChangedFrequency[1],
      props.isToggleChangedFrequency[2],
      !props.isToggleChangedFrequency[3],
      props.isToggleChangedFrequency[4],
    ])
  }

  const handleTurkishVariomeAfToggle = () => {
    props.setToggleSelectorTurkishVariomeAf(!props.toggleSelectorTurkishVariomeAf)
    props.setIsToggleChangedFrequency([
      props.isToggleChangedFrequency[0],
      props.isToggleChangedFrequency[1],
      props.isToggleChangedFrequency[2],
      props.isToggleChangedFrequency[3],
      !props.isToggleChangedFrequency[4],
    ])
  }

  return (
    <Stack direction="column" spacing={2} divider={<Divider />}>
      <YesNoAnyButtonGroup
        title="In DbSNP?"
        value={props.inDbsnpEvent}
        onChange={(value) => props.setInDbsnpEvent(value)}
      />
      <FilterBase
        enabled={props.toggleSelectorAf}
        onToggle={handleAfToggle}
        onChange={(value) => props.setAfEvent(value)}
        value={props.afEvent}
        label="Allele Frequency"
      />
      <FilterBase
        enabled={props.toggleSelectorOneKgFrequency}
        onToggle={handleOneKgToggle}
        onChange={(value) => props.setOneKgFrequencyEvent(value)}
        value={props.oneKgFrequencyEvent}
        label="1KG"
      />
      <FilterBase
        enabled={props.toggleSelectorGnomAdFrequency}
        onToggle={handleGnomAdFrequencyToggle}
        onChange={(value) => props.setGnomAdFrequencyEvent(value)}
        value={props.gnomAdFrequencyEvent}
        label="GnomAD"
      />
      <FilterBase
        enabled={props.toggleSelectorExAcAf}
        onToggle={handleExAcAfToggle}
        onChange={(value) => props.setExAcAfEvent(value)}
        value={props.exAcAfEvent}
        label="ExAC"
      />
      <FilterBase
        enabled={props.toggleSelectorTurkishVariomeAf}
        onToggle={handleTurkishVariomeAfToggle}
        onChange={(value) => props.setTurkishVariomeAfEvent(value)}
        value={props.turkishVariomeAfEvent}
        label="TurkishVariome"
      />
    </Stack>
  )
}

export default FrequencyFilters
