import {
  Autocomplete,
  Button,
  Checkbox,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Divider,
  Stack,
  IconButton,
} from '@material-ui/core'
import Delete from '@material-ui/icons/Delete'
import { useState, useMemo, useEffect } from 'react'
import { GENE_PANEL_DEFAULTS, GENE_CHOICES } from 'src/constants'
import { fetchGeneSet, updateGeneSet, removeGeneSet } from 'src/api/gene'

const sanitizeGeneInput = (input) => {
  const sanitized = input.reduce((acc, o) => {
    if (!o.includes(',')) return acc.length > 0 ? [...acc, o] : [o]
    const genes = o.replace(/\s/g, '').split(',')
    return acc.length > 0 ? [...acc, ...genes] : genes
  }, [])
  return sanitized.filter((o) => GENE_CHOICES.includes(o))
}

function AddSetModal({ open, onClose, uploadGeneSet }) {
  const [name, setName] = useState('')
  const [content, setContent] = useState([])

  const filtered = useMemo(() => {
    return GENE_CHOICES?.filter((o) => ![name].includes(o))
  }, [name])

  const handleNameChange = (event) => {
    setName(event.target.value)
  }

  return (
    <Dialog open={open} onClose={onClose} fullWidth>
      <DialogTitle>Add Gene Set</DialogTitle>
      <DialogContent>
        <TextField
          autoFocus
          margin="dense"
          id="name"
          label="Name"
          type="text"
          fullWidth
          value={name}
          onChange={handleNameChange}
        />
        <Autocomplete
          freeSolo
          multiple
          options={filtered}
          renderInput={(params) => <TextField {...params} label="Genes" variant="outlined" />}
          value={content}
          onChange={(event, newValue) => {
            setContent(sanitizeGeneInput(newValue))
          }}
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} color="primary">
          Cancel
        </Button>
        <Button onClick={() => uploadGeneSet(name, content)} color="primary">
          Add
        </Button>
      </DialogActions>
    </Dialog>
  )
}

function GeneSetFilters({ geneSets, ...props }) {
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [panelOptions, setPanelOptions] = useState([])

  const fetchData = async () => {
    const data = await fetchGeneSet()
    setPanelOptions([...data, ...GENE_PANEL_DEFAULTS])
  }

  useEffect(() => {
    if (panelOptions.length === 0) fetchData()
  }, [panelOptions])

  const uploadGeneSet = (name, content) => {
    updateGeneSet(name, content).then(() => {
      setPanelOptions([...panelOptions, { label: name, value: name, custom: true }])
      props.setGeneSets([...geneSets, { label: name, value: name, custom: true }])
      setIsModalOpen(false)
    })
  }

  const handleGenePanelChange = (event, panel) => {
    props.setGeneSets(panel)
  }

  const deleteGeneSet = (name) => {
    removeGeneSet(name).then(() => {
      fetchData().then(() => {
        props.setGeneSets(geneSets.filter((e) => e.value !== name))
      })
    })
  }

  return (
    <>
      <AddSetModal open={isModalOpen} onClose={() => setIsModalOpen(false)} uploadGeneSet={uploadGeneSet} />
      <Stack direction="column" spacing={2} divider={<Divider />}>
        {geneSets && props.setGeneSets && (
          <Autocomplete
            multiple
            options={panelOptions}
            disableCloseOnSelect
            renderInput={(params) => {
              return <TextField {...params} label="Panels" variant="outlined" />
            }}
            renderOption={(props, option) => (
              <li {...props}>
                <Checkbox checked={geneSets.find((e) => e.value === option.value) !== undefined} />
                {option.label}
                {option.custom && (
                  <IconButton
                    onClick={() => deleteGeneSet(option.value)}
                    sx={{ margin: 'auto 0 auto auto' }}
                    color="error"
                    variant="outlined"
                  >
                    <Delete />
                  </IconButton>
                )}
              </li>
            )}
            getOptionLabel={(option) => option.label}
            value={geneSets}
            onChange={handleGenePanelChange}
          />
        )}
        <Button onClick={() => setIsModalOpen(true)}> Add New Panel</Button>
      </Stack>
    </>
  )
}

export default GeneSetFilters

/*
function ToggleBase({ enabled, onToggle, label }) {
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
    </Grid>
  )
}
*/