import { Checkbox, Divider, Grid, Stack, TextField, Typography } from '@material-ui/core'

function FilterBase({ enabled, onToggle, onChange, value, label }) {
  return (
    <Grid container direction="row" alignItems="center">
      <Grid item xs={1}>
        <Checkbox checked={enabled} onChange={onToggle} />
      </Grid>
      <Grid item xs={4}>
        <Typography variant="body1" color={enabled ? undefined : 'text.disabled'}>
          {label}
        </Typography>
      </Grid>
      <Grid item xs={7}>
        <TextField value={value} onChange={onChange} variant="outlined" fullWidth disabled={!enabled} />
      </Grid>
    </Grid>
  )
}

function LocationFilters(props) {
  const handleChromToggle = () => {
    props.setToggleSelectorChrom(!props.toggleSelectorChrom)
    props.setIsToggleChangedLoc([
      !props.isToggleChangedLoc[0],
      props.isToggleChangedLoc[1],
      props.isToggleChangedLoc[2],
    ])
  }

  const handlePosToggle = () => {
    props.setToggleSelectorPos(!props.toggleSelectorPos)
    props.setIsToggleChangedLoc([
      props.isToggleChangedLoc[0],
      !props.isToggleChangedLoc[1],
      props.isToggleChangedLoc[2],
    ])
  }

  const handleGeneNamesToggle = () => {
    props.setToggleSelectorGene(!props.toggleSelectorGene)
    props.setIsToggleChangedLoc([
      props.isToggleChangedLoc[0],
      props.isToggleChangedLoc[1],
      !props.isToggleChangedLoc[2],
    ])
  }

  return (
    <Stack direction="column" spacing={2} divider={<Divider />}>
      <FilterBase
        label="Chromosome"
        value={props.chromEvent || ''}
        enabled={props.toggleSelectorChrom}
        onChange={(e) => props.setChromEvent(e.target.value)}
        onToggle={handleChromToggle}
      />
      <FilterBase
        label="Position"
        value={props.posEvent || ''}
        enabled={props.toggleSelectorPos}
        onChange={(e) => props.setPosEvent(e.target.value)}
        onToggle={handlePosToggle}
      />
      <FilterBase
        label="Gene Names"
        value={props.geneNamesInputText}
        enabled={props.toggleSelectorGene}
        onToggle={handleGeneNamesToggle}
        onChange={(e) => props.setGeneNamesInputText(e.target.value)}
      />
    </Stack>
  )
}

export default LocationFilters
