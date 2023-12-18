import { Box, Button, Card, Container, Stack, Typography } from '@material-ui/core'
// material
import { styled } from '@material-ui/core/styles'
import { MHidden } from '../../components/@material-extend'
import { LoginForm } from '../../components/authentication/login'
import { LoadingButton } from '@material-ui/lab'
// components
import Page from '../../components/Page'
import Login from './Login'
import LoginAdmin from './LoginAdmin'
import {Link} from '@material-ui/core'
// hooks
// layouts
import AuthLayout from '../../layouts/AuthLayout'
// routes
import { PATH_AUTH } from '../../routes/paths'
import { Link as RouterLink } from 'react-router-dom'
// ----------------------------------------------------------------------

const RootStyle = styled(Page)(({ theme }) => ({
  [theme.breakpoints.up('md')]: {
    display: 'flex',
  },
}));

const SectionStyle = styled(Card)(({ theme }) => ({
  width: '100%',
  maxWidth: 464,
  display: 'flex',
  flexDirection: 'column',
  justifyContent: 'flex-end',
  margin: theme.spacing(2, 0, 2, 2),
  padding: theme.spacing(0, 0, 3),
}));

const ContentStyle = styled('div')(({ theme }) => ({
  maxWidth: 520,
  margin: 'auto',
  display: 'flex',
  minHeight: '100vh',
  flexDirection: 'column',
  justifyContent: 'center',
  padding: theme.spacing(12, 0),
}));

export default function LoginHealthCenter() {
  return (
    <RootStyle title="Login | Priovar">
      <AuthLayout />

      <MHidden width="mdDown">
        <SectionStyle>
          <Typography variant="h3" sx={{ px: 5, mb: 5 }}>
            Hi, Welcome Back
          </Typography>
          <img src="/static/illustrations/illustration_login.png" alt="login" />
        </SectionStyle>
      </MHidden>

      <Container maxWidth="sm">
        <ContentStyle>
          <Stack direction="row" alignItems="center" sx={{ mb: 5 }}>
            <Box sx={{ flexGrow: 1 }}>
              <Typography variant="h2" gutterBottom>
                Sign in to Priovar
              </Typography>
              <Typography sx={{ color: 'text.secondary' }}>Enter your details below.</Typography>
            </Box>
          </Stack>
          <Stack direction="row" alignItems="center" sx={{ mb: 5 }}>
          <Button size="large" color="inherit"  variant="contained" component={RouterLink} to={PATH_AUTH.login} sx={{ mt: 5 }}>
                Clinician Portal
            </Button>
            <Button size="large" color='success' variant="contained" component={RouterLink} to={PATH_AUTH.loginHealthCenter} sx={{ mt: 5 }}>
                Health Center Portal
            </Button>
            <Button size="large" color="inherit"  variant="contained" component={RouterLink} to={PATH_AUTH.loginAdmin} sx={{ mt: 5 }}>
                Admin Portal
            </Button>
          </Stack>
          <LoginForm callerPage={'LoginHealthCenter'}/>
        </ContentStyle>
      </Container>
    </RootStyle>
  );
}