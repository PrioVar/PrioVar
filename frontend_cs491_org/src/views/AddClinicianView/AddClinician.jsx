import closeFill from '@iconify/icons-eva/close-fill'
import eyeFill from '@iconify/icons-eva/eye-fill'
import eyeOffFill from '@iconify/icons-eva/eye-off-fill'
import { Icon } from '@iconify/react'
import axios from 'axios'
// material
import {
  Alert,
  Checkbox,
  FormControlLabel,
  IconButton,
  InputAdornment,
  Link,
  Stack,
  TextField,
  Box,
  Typography,
  Container
} from '@material-ui/core'
import { LoadingButton } from '@material-ui/lab'
import { Form, FormikProvider, useFormik } from 'formik'
import { useSnackbar } from 'notistack5'
import { useState } from 'react'
import { Link as RouterLink } from 'react-router-dom'
import * as Yup from 'yup'
// hooks
import useIsMountedRef from '../../hooks/useIsMountedRef'
// routes
import { PATH_AUTH, PATH_PrioVar, ROOTS_PrioVar } from '../../routes/paths'
//
import { MIconButton } from '../../components/@material-extend'
//
// ----------------------------------------------------------------------

export default function AddNewClinician() {
  const isMountedRef = useIsMountedRef()
  const { enqueueSnackbar, closeSnackbar } = useSnackbar()
  const [showPassword, setShowPassword] = useState(false)
  const healthCenterId = localStorage.getItem('healthCenterId') || '';

  const AddClinicianSchema = Yup.object().shape({
    name: Yup.string().required('Name is required'),
    email: Yup.string().email('Email must be a valid email address').required('Email is required'),
    password: Yup.string().required('Password is required'),
  })

  const formik = useFormik({
    initialValues: {
      name: '',
      email: '',
      password: '',
      remember: true,
    },
    //validationSchema: LoginSchema, uncomment this line to enable validation
    onSubmit: async (values, { setErrors, setSubmitting, resetForm }) => {
      try {
        // get the email from formik
        // get the password from formik
        const namePrioVar = values.name
        const emailPrioVar = values.email
        const passwordPrioVar = values.password
        const clinician = {
            name: namePrioVar,
            email: emailPrioVar,
            password: passwordPrioVar,
            medicalCenter: {
                id: healthCenterId
            }
        }

        const { data } = await axios.post(`${ROOTS_PrioVar}/clinician/add`, clinician)
        enqueueSnackbar('Clinician Added Successfully!', {
            variant: 'success',
            action: (key) => (
                <MIconButton size="small" onClick={() => {
                  closeSnackbar(key);
                  // Reload the page after the request is successful
                  window.location.reload();
                }}>
                  <Icon icon={closeFill} />
                </MIconButton>
              ),
          })
      }
      catch (error) {
        // print the error details, response is a JSON and has field 'message'
        enqueueSnackbar(error.response.data.message, {
            variant: 'error',
            action: (key) => (
              <MIconButton size="small" onClick={() => closeSnackbar(key)}>
                <Icon icon={closeFill} />
              </MIconButton>
            ),
        })
        console.error(error)
        resetForm()
        if (isMountedRef.current) {
          setSubmitting(false)
          setErrors({ afterSubmit: error.response.data?.non_field_errors?.[0] ?? error.message })
        }
      }
    },
  })

  const { errors, touched, values, isSubmitting, handleSubmit, getFieldProps } = formik

  const handleShowPassword = () => {
    setShowPassword((show) => !show)
  }

  return (
    <Container maxWidth="md">
        <Box p={4} bgcolor="background.default" display="flex" flexDirection="column" justifyContent="center" alignItems="center" width="%50">
            <Typography variant="h4" gutterBottom align="center" mt={4}>
            Adding A New Clinician
            </Typography>
            <FormikProvider value={formik}>
            <Form autoComplete="off" noValidate onSubmit={handleSubmit}>
                <Stack spacing={3}>
                {errors.afterSubmit && <Alert severity="error">{errors.afterSubmit}</Alert>}

                <TextField
                    fullWidth
                    autoComplete="name"
                    type="text"
                    label="Name"
                    {...getFieldProps('name')}
                    error={Boolean(touched.name && errors.name)}
                    helperText={touched.name && errors.name}
                />

                <TextField
                    fullWidth
                    autoComplete="username"
                    type="email"
                    label="Email address"
                    {...getFieldProps('email')}
                    error={Boolean(touched.email && errors.email)}
                    helperText={touched.email && errors.email}
                />

                <TextField
                    fullWidth
                    autoComplete="current-password"
                    type={showPassword ? 'text' : 'password'}
                    label="Password"
                    {...getFieldProps('password')}
                    InputProps={{
                    endAdornment: (
                        <InputAdornment position="end">
                        <IconButton onClick={handleShowPassword} edge="end">
                            <Icon icon={showPassword ? eyeFill : eyeOffFill} />
                        </IconButton>
                        </InputAdornment>
                    ),
                    }}
                    error={Boolean(touched.password && errors.password)}
                    helperText={touched.password && errors.password}
                />
                </Stack>
                
                <Box mt={2}>
                    <LoadingButton fullWidth size="large" type="submit" variant="contained" loading={isSubmitting}>
                        Add
                    </LoadingButton>
                </Box>
            </Form>
            </FormikProvider>
        </Box>
    </Container>
  )
}
