import React from 'react'
import { Typography, Grid, Checkbox, Stack, Divider } from '@material-ui/core'
import MinMaxField from 'src/views/VariantsView/Toolbar/FiltersPanelMenu/MinMaxField'

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

function ReadFilters(props) {
  const handleDpToggle = () => {
    props.setDpFilterToggle(!props.dpFilterToggle)
  }

  const handleABToggle = () => {
    props.setAbFilterToggle(!props.abFilterToggle)
  }
  return (
    <Stack direction="column" spacing={2} divider={<Divider />}>
      <FilterBase
        label="Read Depth (DP)"
        min={0}
        max={1000}
        value={props.dpFilter}
        enabled={props.dpFilterToggle}
        onToggle={handleDpToggle}
        onChange={(value) => props.setDpFilter(value)}
      />
      <FilterBase
        label="Allelic Balance (AB)"
        min={0}
        max={1000}
        value={props.abFilter}
        enabled={props.abFilterToggle}
        onToggle={handleABToggle}
        onChange={(value) => props.setAbFilter(value)}
      />
    </Stack>
  )
}

export default ReadFilters
