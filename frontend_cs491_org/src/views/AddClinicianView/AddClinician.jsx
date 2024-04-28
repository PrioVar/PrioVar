import closeFill from '@iconify/icons-eva/close-fill'
import eyeFill from '@iconify/icons-eva/eye-fill'
import eyeOffFill from '@iconify/icons-eva/eye-off-fill'
import { Icon } from '@iconify/react'
import axios from 'axios'
// material
import {
  Alert,
  IconButton,
  InputAdornment,
  Stack,
  TextField,
  Box,
  Typography,
  Container,
  Grid,
  TableCell,
  TableRow,
  Table,
  TableHead,
  TableBody,
  TableContainer,   
} from '@material-ui/core'
import { LoadingButton } from '@material-ui/lab'
import { Form, FormikProvider, useFormik } from 'formik'
import { useSnackbar } from 'notistack5'
import { useState, useEffect } from 'react'
import * as Yup from 'yup'
// hooks
import useIsMountedRef from '../../hooks/useIsMountedRef'
// routes
import { ROOTS_PrioVar } from '../../routes/paths'
//
import { MIconButton } from '../../components/@material-extend'
import Page from 'src/components/Page'

//
// ----------------------------------------------------------------------

export default function AddNewClinician() {
  const isMountedRef = useIsMountedRef()
  const { enqueueSnackbar, closeSnackbar } = useSnackbar()
  const [showPassword, setShowPassword] = useState(false)
  const healthCenterId = localStorage.getItem('healthCenterId') || '';

  const [details, setDetails] = useState(
    []
  );

  useEffect(() => {
    const fetchCliniciansAndPatients = async () => {
      try {
        const clinicianResponse = await axios.get(`${ROOTS_PrioVar}/clinician/byMedicalCenter/${healthCenterId}`);
        const clinicians = clinicianResponse.data;
        console.log("SUCCESS", clinicians);
  
        const patientCountsPromises = clinicians.map(clinician =>
          axios.get(`${ROOTS_PrioVar}/clinician/allPatients/${clinician.id}`)
            .then(response => ({ clinicianId: clinician.id, patientCount: response.data.length }))
        );
  
        const patientCounts = await Promise.all(patientCountsPromises);
  
        const detailedClinicians = clinicians.map(clinician => {
          const patientInfo = patientCounts.find(info => info.clinicianId === clinician.id);
          return { ...clinician, patientCount: patientInfo ? patientInfo.patientCount : 0 };
        });
  
        if (isMountedRef.current) {
          setDetails(detailedClinicians);
        }
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };
  
    fetchCliniciansAndPatients();
  }, [healthCenterId, isMountedRef]);

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
        let errorMessage = 'An unexpected error occurred';
        if (error.response) {
            // The request was made and the server responded with a status code
            // that falls out of the range of 2xx, and the response has a direct string body.
            errorMessage = error.response.data || error.message;
        } else if (error.request) {
            // The request was made but no response was received
            errorMessage = 'No response received from the server';
        } else {
            // Something happened in setting up the request that triggered an Error
            errorMessage = error.message;
        }
        // print the error details, response is a JSON and has field 'message'
        enqueueSnackbar(errorMessage, {
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
    <Page  style={{
        position: 'absolute', 
        top: 0, 
        left: 0, 
        width: '100%', 
        height: '100vh', 
        backgroundImage: 'url("/static/new_images/dna3.png")', 
        backgroundSize: 'cover', 
        backgroundPosition: 'center center',
        backgroundColor: 'rgba(255, 255, 255, 0.7)', // Adds white transparency
        backgroundBlendMode: 'overlay' // This blends the background color with the image
      }}>
        <Grid container spacing={3} sx={{mt:4}}>
            {/* Left Half: Table (adjust the content of this part as needed) */}
            <Grid item xs={6}>
                <Box p={2} style={{ marginTop: '50px' }}>
                    <Typography variant="h4" gutterBottom align="center">
                    Clinicians
                    </Typography>
                    <TableContainer>
                    <Table>
                        <TableHead>
                        <TableRow>
                            <TableCell>ID</TableCell>
                            <TableCell>Name</TableCell>
                            <TableCell>Email</TableCell>
                            <TableCell>Patient Count</TableCell> {/* Added header for patient count */}
                        </TableRow>
                        </TableHead>
                        <TableBody>
                        {/* Add dummy table rows */}
                        {details ? details.map((row, index) => (
                                <TableRow key={index + 1}>
                                <TableCell>{index + 1}</TableCell>
                                <TableCell>{row?.name}</TableCell>
                                <TableCell>{row?.email}</TableCell>
                                <TableCell>{row?.patientCount}</TableCell>
                                {/* Add more table cells as needed */}
                                </TableRow>
                            )): null}
                        </TableBody>
                    </Table>
                    </TableContainer>
                </Box>
            </Grid>

            <Grid item xs={6}>
                <Box p={4}  display="flex" flexDirection="column" justifyContent="center" alignItems="center" width="%50">
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
                            InputLabelProps={{
                                style: { color: 'black' }
                              }}
                              InputProps={{
                                style: { color: 'black' },
                              }}
                        />

                        <TextField
                            fullWidth
                            autoComplete="username"
                            type="email"
                            label="Email address"
                            {...getFieldProps('email')}
                            error={Boolean(touched.email && errors.email)}
                            helperText={touched.email && errors.email}
                            InputLabelProps={{
                                style: { color: 'black' }
                              }}
                              InputProps={{
                                style: { color: 'black' },
                              }}
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
                                style: { color: 'black' }
                              }}
                              InputLabelProps={{
                                style: { color: 'black' }
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
            </Grid>
        </Grid>
    </Page>
  )
}
