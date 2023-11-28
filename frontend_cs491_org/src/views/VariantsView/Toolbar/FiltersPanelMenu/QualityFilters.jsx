import React from 'react'
import { Typography, Grid, Checkbox, Stack, Divider } from '@material-ui/core'
import MinMaxField from 'src/views/VariantsView/Toolbar/FiltersPanelMenu/MinMaxField'
import YesNoAnyButtonGroup from 'src/views/VariantsView/Toolbar/FiltersPanelMenu/YesNoAnyButtonGroup'

function FilterBase({ enabled, onToggle, onChange, value, label, min, max }) {
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
        <MinMaxField title={label} value={value} onChange={onChange} min={min} max={max} disabled={!enabled} />
      </Grid>
    </Grid>
  )
}

function QualityFilters(props) {
  const handleQualFToggle = () => {
    props.setToggleSelectorQualF(!props.toggleSelectorQualF)
    props.setIsToggleChangedQuality([!props.isToggleChangedQuality[0], props.isToggleChangedQuality[1]])
  }

  const handleFsFToggle = () => {
    props.setToggleSelectorFsF(!props.toggleSelectorFsF)
    props.setIsToggleChangedQuality([props.isToggleChangedQuality[0], !props.isToggleChangedQuality[1]])
  }

  return (
    <Stack direction="column" spacing={2} divider={<Divider />}>
      <YesNoAnyButtonGroup
        title="Variant Calling Filter"
        value={props.filterFEvent}
        onChange={(value) => props.setfilterFEvent(value)}
        copy={{
          any: 'All',
          yes: 'Pass',
          no: 'Fail',
        }}
      />
      <FilterBase
        label="Quality score (QUAL)"
        min={0}
        max={150000}
        value={props.qualFEvent}
        enabled={props.toggleSelectorQualF}
        onToggle={handleQualFToggle}
        onChange={(value) => props.setQuallFEvent(value)}
      />
      <FilterBase
        label="Strand Bias (FS)"
        min={0}
        max={1000}
        value={props.fsFEvent}
        enabled={props.toggleSelectorFsF}
        onToggle={handleFsFToggle}
        onChange={(value) => props.setFsFEvent(value)}
      />
    </Stack>
  )
}

export default QualityFilters
