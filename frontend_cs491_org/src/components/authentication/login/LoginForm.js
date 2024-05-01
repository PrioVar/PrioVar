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
} from '@material-ui/core'
import { LoadingButton } from '@material-ui/lab'
import { Form, FormikProvider, useFormik } from 'formik'
import { useSnackbar } from 'notistack5'
import { useState } from 'react'
import { Link as RouterLink, useNavigate } from 'react-router-dom'
//import * as Yup from 'yup'
// hooks
//import useAuth from '../../../hooks/useAuth'
import useIsMountedRef from '../../../hooks/useIsMountedRef'
// routes
import { PATH_AUTH, ROOTS_PrioVar, PATH_DASHBOARD } from '../../../routes/paths'
//
import { MIconButton } from '../../@material-extend'
//
//const emailFix = "erkinaydin@morpheus.cs.bilkent.edu.tr"
//const passwordFix = "erkinaydin"
// ----------------------------------------------------------------------

export default function LoginForm({ callerPage }) {
  // const { login } = useAuth()
  localStorage.setItem('clinicianId', -1)
  localStorage.setItem('healthCenterId', -1)
  localStorage.setItem('adminId', -1)
  const isMountedRef = useIsMountedRef()
  const { enqueueSnackbar, closeSnackbar } = useSnackbar()
  const [showPassword, setShowPassword] = useState(false)
  const navigate = useNavigate()
  /*
  const LoginSchema = Yup.object().shape({
    email: Yup.string().email('Email must be a valid email address').required('Email is required'),
    password: Yup.string().required('Password is required'),
  })
  */
  const formik = useFormik({
    initialValues: {
      email: '',
      password: '',
      remember: true,
    },
    //validationSchema: LoginSchema, uncomment this line to enable validation
    onSubmit: async (values, { setErrors, setSubmitting, resetForm }) => {
      /*
      try {
        //await login(values.email, values.password)
        await login(emailFix, passwordFix)
        enqueueSnackbar('Login success', {
          variant: 'success',
          action: (key) => (
            <MIconButton size="small" onClick={() => closeSnackbar(key)}>
              <Icon icon={closeFill} />
            </MIconButton>
          ),
        })
        if (isMountedRef.current) {
          setSubmitting(false)
        }
      } catch (error) {
        console.error(error)
        resetForm()
        if (isMountedRef.current) {
          setSubmitting(false)
          setErrors({ afterSubmit: error.response.data?.non_field_errors?.[0] ?? error.message })
        }
      }*/
      try {
        // get the email from formik
        // get the password from formik
        const emailPrioVar = values.email
        const passwordPrioVar = values.password

        var clinicianId = -1
        var healthCenterId = -1
        var adminId = -1
        if(callerPage === 'Login') {
          const { data } = await axios.post(`${ROOTS_PrioVar}/clinician/login?email=${emailPrioVar}&password=${passwordPrioVar}`)
          //console.log(data)
          clinicianId = data.id
          healthCenterId = data.relatedId
        }
        else if(callerPage === 'LoginHealthCenter') {
          const { data } = await axios.post(`${ROOTS_PrioVar}/medicalCenter/login?email=${emailPrioVar}&password=${passwordPrioVar}`)
          healthCenterId = data.id
        }
        else if(callerPage === 'LoginAdmin') {
          const { data } = await axios.post(`${ROOTS_PrioVar}/admin/login?email=${emailPrioVar}&password=${passwordPrioVar}`)
          adminId = data.id
        }
        
        localStorage.setItem('clinicianId', clinicianId)
        localStorage.setItem('healthCenterId', healthCenterId)
        localStorage.setItem('adminId', adminId)
        navigate(PATH_DASHBOARD.root)
        enqueueSnackbar('Login Success!', {
            variant: 'success',
            action: (key) => (
              <MIconButton size="small" onClick={() => closeSnackbar(key)}>
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
    <FormikProvider value={formik}>
      <Form autoComplete="off" noValidate onSubmit={handleSubmit}>
        <Stack spacing={3}>
          {errors.afterSubmit && <Alert severity="error">{errors.afterSubmit}</Alert>}

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

        <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ my: 2 }}>
          <FormControlLabel
            control={<Checkbox {...getFieldProps('remember')} checked={values.remember} />}
            label="Remember me"
          />

          <Link component={RouterLink} variant="subtitle2" to={PATH_AUTH.resetPassword}>
            Forgot password?
          </Link>
        </Stack>

        <LoadingButton fullWidth size="large" type="submit" variant="contained" loading={isSubmitting}>
          Login
        </LoadingButton>
      </Form>
    </FormikProvider>
  )
}
