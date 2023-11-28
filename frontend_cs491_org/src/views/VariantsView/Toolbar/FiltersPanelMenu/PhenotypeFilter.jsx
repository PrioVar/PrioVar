import { Autocomplete, Checkbox, TextField, Divider, Stack, Grid } from '@material-ui/core'
import { DISEASE_CHOICES } from 'src/constants'
import { useState, useMemo } from 'react'

function PhenotypeFilter({ phenotypes, ...props }) {
  const [input, setInput] = useState('')
  const filtered = useMemo(() => {
    if (input.length === 0) return DISEASE_CHOICES['A']['B']
    if (input.length === 1) {
      return DISEASE_CHOICES[input] ? DISEASE_CHOICES[input] : []
    } else if (input.length === 2) {
      const first = input.substring(0, 1).toUpperCase()
      const second = input.substring(1, 2).toUpperCase()
      return DISEASE_CHOICES[first][second] ? DISEASE_CHOICES[first][second] : []
    }
    const first = input.substring(0, 1).toUpperCase()
    const second = input.substring(1, 2).toUpperCase()
    return DISEASE_CHOICES[first][second]?.filter((o) => ![input].includes(o))
  }, [input])
  return (
    <Stack direction="column" spacing={2} divider={<Divider />}>
      {phenotypes && props.setPhenotypes && (
        <Autocomplete
          multiple
          options={filtered}
          disableCloseOnSelect
          renderInput={(params) => {
            return (
              <Grid sx={{ display: 'flex' }}>
                <TextField {...params} label="Diseases" variant="outlined" />
              </Grid>
            )
          }}
          renderOption={(props, option) => (
            <li {...props}>
              <Checkbox checked={phenotypes.find((e) => e.key === option.key) !== undefined} />
              {option.disease}
            </li>
          )}
          getOptionLabel={(option) => option.disease}
          inputValue={input}
          onInputChange={(_, val) => setInput(val)}
          value={phenotypes}
          onChange={(_, val) => props.setPhenotypes(val)}
        />
      )}
    </Stack>
  )
}

export default PhenotypeFilter
