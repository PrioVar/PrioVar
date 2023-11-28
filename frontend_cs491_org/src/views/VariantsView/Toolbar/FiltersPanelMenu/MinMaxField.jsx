import React, { useEffect, useMemo } from 'react'
import PropTypes from 'prop-types'
import { useFormik } from 'formik'
import { Grid, TextField } from '@material-ui/core'
import useDebouncedValue from 'src/hooks/useDebouncedValue'
import * as Yup from 'yup'

function MinMaxField({ title, min = 0, max = 1, onChange, value, disabled, ...rest }) {
  const validationSchema = useMemo(
    () =>
      Yup.object().shape(
        {
          min: Yup.number()
            .label(`Minimum ${title}`)
            .min(min)
            .max(max)
            .when('max', (max, schema) => (max !== undefined ? schema.max(max) : schema))
            .required(),
          max: Yup.number()
            .label(`Maximum ${title}`)
            .min(min)
            .max(max)
            .when('min', (min, schema) => (min !== undefined ? schema.min(min) : schema))
            .required(),
        },
        ['min', 'max'],
      ),
    [max, min, title],
  )

  const formik = useFormik({
    initialValues: {
      min: value[0],
      max: value[1],
    },
    validationSchema,
    validateOnChange: true,
  })

  const debouncedValues = useDebouncedValue(formik.values, 100)

  useEffect(() => {
    // Due to formik bug we have to valdiate twice
    if (validationSchema.isValidSync(debouncedValues)) {
      onChange([debouncedValues.min, debouncedValues.max])
    }
  }, [debouncedValues])

  return (
    // <Stack direction="column" py={1}>
    //   <Typography gutterBottom>{title}</Typography>
    //   <Box p={0.5} />
    <Grid container item spacing={2}>
      <Grid item xs>
        <TextField
          name="min"
          label="Minimum"
          type="number"
          fullWidth
          onChange={formik.handleChange}
          value={formik.values.min}
          error={!!formik.errors.min}
          helperText={formik.errors.min}
          disabled={disabled}
        />
      </Grid>
      <Grid item xs>
        <TextField
          name="max"
          label="Maximum"
          type="number"
          fullWidth
          onChange={formik.handleChange}
          value={formik.values.max}
          error={!!formik.errors.max}
          helperText={formik.errors.max}
          disabled={disabled}
        />
      </Grid>
    </Grid>
    // </Stack>
  )
}
MinMaxField.propTypes = {
  title: PropTypes.string.isRequired,
  min: PropTypes.number,
  max: PropTypes.number,
  onChange: PropTypes.func.isRequired,
}

export default MinMaxField
