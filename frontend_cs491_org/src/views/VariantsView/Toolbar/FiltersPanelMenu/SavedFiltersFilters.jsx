import {
  Autocomplete,
  Button,
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
import { useState, useEffect } from 'react'
import { fetchSavedFilters, updateSavedFilter, removeSavedFilter } from 'src/api/filters'

function AddSetModal({ open, onClose, uploadFilter }) {
  const [name, setName] = useState('')

  const handleNameChange = (event) => {
    setName(event.target.value)
  }

  return (
    <Dialog open={open} onClose={onClose} fullWidth>
      <DialogTitle>Save Filter</DialogTitle>
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
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} color="primary">
          Cancel
        </Button>
        <Button onClick={() => uploadFilter(name)} color="primary">
          Add
        </Button>
      </DialogActions>
    </Dialog>
  )
}

function SavedFilters({ chosenFilter, setChosenFilter, handleSave }) {
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [savedFilters, setSavedFilters] = useState([])

  useEffect(() => {
    const fetchData = async () => {
      const data = await fetchSavedFilters()
      setSavedFilters(data)
    }
    fetchData()
  }, [setSavedFilters])

  const uploadFilter = (name) => {
    const state = handleSave(true)
    if (!state) return
    updateSavedFilter(name, state).then(() => {
      setSavedFilters([...savedFilters, { label: name, value: state }])
      setIsModalOpen(false)
    })
  }

  const deleteSavedFilter = (name) => {
    removeSavedFilter(name).then(() => {
      setSavedFilters(savedFilters.filter((e) => e.name !== name))
    })
  }

  return (
    <>
      <AddSetModal open={isModalOpen} onClose={() => setIsModalOpen(false)} uploadFilter={uploadFilter} />
      <Stack direction="column" spacing={2} divider={<Divider />}>
        {savedFilters && setSavedFilters && (
          <Autocomplete
            options={savedFilters}
            renderInput={(params) => {
              return <TextField {...params} label="Saved Filters" variant="outlined" />
            }}
            renderOption={(props, option) => (
              <li {...props}>
                {option.label}
                <IconButton
                  onClick={() => deleteSavedFilter(option.name)}
                  sx={{ margin: 'auto 0 auto auto' }}
                  color="error"
                  variant="outlined"
                >
                  <Delete />
                </IconButton>
              </li>
            )}
            getOptionLabel={(option) => option.name}
            value={chosenFilter}
            onChange={(_e, newFilter) => setChosenFilter(newFilter)}
          />
        )}
        <Button onClick={() => setIsModalOpen(true)}>Save Current Filter</Button>
      </Stack>
    </>
  )
}

export default SavedFilters
