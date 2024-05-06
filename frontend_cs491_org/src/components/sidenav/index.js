import { useState, useEffect } from 'react'
import { Icon } from '@iconify/react'
import closeFill from '@iconify/icons-eva/close-fill'
import options2Fill from '@iconify/icons-eva/options-2-fill'
// material
import { Box, Button, Backdrop, Paper, Tooltip, Divider, Typography, Stack } from '@material-ui/core'
//
import Scrollbar from '../Scrollbar'
import { MIconButton } from '../@material-extend'


import { PATH_DASHBOARD } from '../../routes/paths'
import { Link as RouterLink } from 'react-router-dom'
// ----------------------------------------------------------------------

const DRAWER_WIDTH = 260

export default function SideNav() {
  const [open, setOpen] = useState(false)
  // Assuming you have access to localStorage and handleClose function
  const clinicianId = parseInt(localStorage.getItem('clinicianId'));

  useEffect(() => {
    if (open) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = 'unset'
    }
  }, [open])

  const handleToggle = () => {
    setOpen((prev) => !prev)
  }

  const handleClose = () => {
    setOpen(false)
  }

  return (
    <>
      <Backdrop sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }} open={open} onClick={handleClose} />

      <Box
        sx={{
          top: 12,
          bottom: 12,
          left: 0,
          position: 'fixed',
          zIndex: 2001,
          ...(open && { left: 12 }), // change this part to make it left aligned? maybe??
        }}
      >
        <Box
          sx={{
            p: 0.5,
            px: '4px',
            mt: -3,
            right: -44,
            top: '50%',
            color: 'grey.800',
            position: 'absolute',
            bgcolor: 'common.white',
            borderRadius: '24px 16px 16px 24px',
            boxShadow: (theme) => theme.customShadows.z12,
          }}
        >
          <Tooltip title="Side Navigation">
            <MIconButton
              color="inherit"
              onClick={handleToggle}
              sx={{
                p: 0,
                width: 40,
                height: 40,
                transition: (theme) => theme.transitions.create('all'),
                '&:hover': { color: 'primary.main', bgcolor: 'transparent' },
              }}
            >
              <Icon icon={open ? closeFill : options2Fill} width={20} height={20} />
            </MIconButton>
          </Tooltip>
        </Box>

        <Paper
          sx={{
            height: 1,
            width: '0px',
            overflow: 'hidden',
            boxShadow: (theme) => theme.customShadows.z24,
            transition: (theme) => theme.transitions.create('width'),
            ...(open && { width: DRAWER_WIDTH }),
          }}
        >
          <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ py: 2, pr: 1, pl: 2.5 }}>
            <Typography variant="subtitle1">Navigate</Typography>
            <MIconButton onClick={handleClose}>
              <Icon icon={closeFill} width={20} height={20} />
            </MIconButton>
          </Stack>
          <Divider />

          <Scrollbar sx={{ height: 1 }}>
            <Stack spacing={4} sx={{ pt: 3, px: 3, pb: 15 }}>
              <Stack direction="column" spacing={1.5}>
              <Button size="large" color='inherit' variant="contained" component={RouterLink} onClick={handleClose} to={PATH_DASHBOARD.general.files} sx={{ mt: 5 }}>
                Dashboard
              </Button>
              {clinicianId !== -1 && (
                <Button size="large" color="inherit" variant="contained" component={RouterLink} onClick={handleClose} to={PATH_DASHBOARD.general.myPatients} sx={{ mt: 5 }}>
                  My Patients
                </Button>
              )}
              <Button size="large" color="inherit" variant="contained" component={RouterLink} onClick={handleClose} to={PATH_DASHBOARD.general.clinicPatients} sx={{ mt: 5 }}>
                  Clinics Patients
              </Button>
              <Button size="large" color="inherit" variant="contained" component={RouterLink} onClick={handleClose} to={PATH_DASHBOARD.general.requestedPatients} sx={{ mt: 5 }}>
                  Cross-Clinic Patients
              </Button>
              <Button size="large" color="inherit" variant="contained" component={RouterLink} onClick={handleClose} to={PATH_DASHBOARD.general.customQuery} sx={{ mt: 5 }}>
                  Search Population
              </Button>
              {/**TODO */}
              {clinicianId === -1 && (
                <Button size="large" color="inherit" variant="contained" component={RouterLink} onClick={handleClose} to={PATH_DASHBOARD.general.subscriptionPlans} sx={{ mt: 5 }}>
                  Subscription Plans
                </Button>
              )}
              <Button size="large" color="inherit" variant="contained" component={RouterLink} onClick={handleClose} to={PATH_DASHBOARD.general.aiSupport} sx={{ mt: 5 }}>
                  AI Support
              </Button>
              <Button size="large" color="inherit" variant="contained" component={RouterLink} onClick={handleClose} to={PATH_DASHBOARD.general.informationRetrieval} sx={{ mt: 5 }}>
                  AI Information Retrieval
              </Button>
              {clinicianId === -1 && (
                <Button size="large" color="inherit" variant="contained" component={RouterLink} onClick={handleClose} to={PATH_DASHBOARD.general.addClinician} sx={{ mt: 5 }}>
                  Manage Clinicians
                </Button>
              )}
              </Stack>

              {/*              <Stack spacing={1.5}>
                <Typography variant="subtitle2">Direction</Typography>
                <SettingDirection />
              </Stack>*/}
              {/*
              <Stack spacing={1.5}>
                <Typography variant="subtitle2">Color</Typography>
                <SettingColor />
              </Stack>*/}
              {/*
              <Stack spacing={1.5}>
                <Typography variant="subtitle2">Stretch</Typography>
                <SettingStretch />
              </Stack>*/}
            </Stack>
          </Scrollbar>
        </Paper>
      </Box>
    </>
  )
}
