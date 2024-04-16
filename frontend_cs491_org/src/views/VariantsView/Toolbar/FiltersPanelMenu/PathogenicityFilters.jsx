import React from 'react'
import { Checkbox, Divider, Grid, Stack, Typography } from '@material-ui/core'
import MinMaxField from 'src/views/VariantsView/Toolbar/FiltersPanelMenu/MinMaxField'
import InfoIcon from '@mui/icons-material/Info'
import IconButton from '@mui/material/IconButton'
import ToolTip from '@mui/material/Tooltip'

function FilterBase({ enabled, onToggle, onChange, value, label, minMax = true, tooltip }) {
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
      {minMax && (
        <Grid sx={{ display: 'flex' }} item xs={6}>
          <MinMaxField title={label} value={value} onChange={onChange} disabled={!enabled} />
          {tooltip && (
            <ToolTip title={tooltip}>
              <IconButton>
                <InfoIcon />
              </IconButton>
            </ToolTip>
          )}
        </Grid>
      )}
    </Grid>
  )
}

function PathogenicityFilters(props) {
  const handleCaddToggle = () => {
    props.setToggleSelectorCadd(!props.toggleSelectorCadd)
    props.setIsToggleChangedPatho([
      !props.isToggleChangedPatho[0],
      props.isToggleChangedPatho[1],
      props.isToggleChangedPatho[2],
      props.isToggleChangedPatho[3],
      props.isToggleChangedPatho[4],
      props.isToggleChangedPatho[5],
    ])
  }

  const handleDannToggle = () => {
    props.setToggleSelectorDann(!props.toggleSelectorDann)
    props.setIsToggleChangedPatho([
      props.isToggleChangedPatho[0],
      !props.isToggleChangedPatho[1],
      props.isToggleChangedPatho[2],
      props.isToggleChangedPatho[3],
      props.isToggleChangedPatho[4],
      props.isToggleChangedPatho[5],
    ])
  }

  const handleMetalRToggle = () => {
    props.setToggleSelectorMetalR(!props.toggleSelectorMetalR)
    props.setIsToggleChangedPatho([
      props.isToggleChangedPatho[0],
      props.isToggleChangedPatho[1],
      !props.isToggleChangedPatho[2],
      props.isToggleChangedPatho[3],
      props.isToggleChangedPatho[4],
      props.isToggleChangedPatho[5],
    ])
  }

  const handlePolyToggle = () => {
    props.setToggleSelectorPoly(!props.toggleSelectorPoly)
    props.setIsToggleChangedPatho([
      props.isToggleChangedPatho[0],
      props.isToggleChangedPatho[1],
      props.isToggleChangedPatho[2],
      !props.isToggleChangedPatho[3],
      props.isToggleChangedPatho[4],
      props.isToggleChangedPatho[5],
    ])
  }

  const handleSiftToggle = () => {
    props.setToggleSelectorSift(!props.toggleSelectorSift)
    props.setIsToggleChangedPatho([
      props.isToggleChangedPatho[0],
      props.isToggleChangedPatho[1],
      props.isToggleChangedPatho[2],
      props.isToggleChangedPatho[3],
      !props.isToggleChangedPatho[4],
      props.isToggleChangedPatho[5],
    ])
  }

  return (
    <Stack direction="column" spacing={2} divider={<Divider />}>
      <FilterBase
        enabled={props.toggleSelectorCadd}
        onToggle={handleCaddToggle}
        onChange={(value) => props.setCaddEvent(value)}
        value={props.caddEvent}
        label="CADD"
        tooltip='Rentzsch, Philipp, et al. "CADD: predicting the deleteriousness of variants throughout the human genome." Nucleic acids research 47.D1 (2019): D886-D894.'
      />
      <FilterBase
        enabled={props.toggleSelectorDann}
        onToggle={handleDannToggle}
        onChange={(value) => props.setDannEvent(value)}
        value={props.dannEvent}
        label="DANN"
        tooltip='Quang, Daniel, Yifei Chen, and Xiaohui Xie. "DANN: a deep learning approach for annotating the pathogenicity of genetic variants." Bioinformatics 31.5 (2015): 761-763.
        '
      />
      <FilterBase
        enabled={props.toggleSelectorMetalR}
        onToggle={handleMetalRToggle}
        onChange={(value) => props.setMetalREvent(value)}
        value={props.metalREvent}
        label="MetalR"
        tooltip='Chen, Yixiong, et al. "Metalr: Layer-wise learning rate based on meta-learning for adaptively fine-tuning medical pre-trained models." arXiv preprint arXiv:2206.01408 (2022).'
      />
      <FilterBase
        enabled={props.toggleSelectorPoly}
        onToggle={handlePolyToggle}
        onChange={(value) => props.setPolyEvent(value)}
        value={props.polyEvent}
        label="PolyPhen"
        tooltip='Adzhubei, Ivan A., et al. "A method and server for predicting damaging missense mutations." Nature methods 7.4 (2010): 248-249.'
      />
      <FilterBase
        enabled={props.toggleSelectorSift}
        onToggle={handleSiftToggle}
        onChange={(value) => props.setSiftEvent(value)}
        value={props.siftEvent}
        label="SIFT"
        tooltip="Ng, Pauline C., and Steven Henikoff. 'SIFT: Predicting amino acid changes that affect protein function.' Nucleic acids research 31.13 (2003): 3812-3814."
      />
    </Stack>
  )
}

export default PathogenicityFilters
