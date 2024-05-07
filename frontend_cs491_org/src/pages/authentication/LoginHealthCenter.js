import { Box, Button, Card, Container, Stack, Typography, useMediaQuery, useTheme } from '@material-ui/core'
// material
import { styled } from '@material-ui/core/styles'
import { LoginForm } from '../../components/authentication/login'

// components
import Page from '../../components/Page'
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
  const theme = useTheme();
  const isSmallScreen = useMediaQuery(theme.breakpoints.down('sm'));
  return (
    <RootStyle title="Login | PrioVar">
      <AuthLayout />

      <Container maxWidth="sm">
        <ContentStyle>
          <Stack direction="row" alignItems="center" sx={{ mb: 5 }}>
            <Box sx={{ flexGrow: 1 }}>
              <Typography variant="h2" gutterBottom>
                Sign in to PrioVar
              </Typography>
              <Typography sx={{ color: 'text.secondary' }}>Enter your details below.</Typography>
            </Box>
          </Stack>
          <Stack direction={isSmallScreen ? 'column' : 'row'} alignItems={isSmallScreen ? 'center' : 'flex-start'} sx={{ mb: 5 }}>
            <Button
              size="large"
              color="inherit"
              variant="contained"
              component={RouterLink}
              to={PATH_AUTH.login}
              sx={{ mt: isSmallScreen ? 2 : 0, mr: isSmallScreen ? 0 : 2 }}
            >
              Clinician Portal
            </Button>
            <Button
              size="large"
              color="success"
              variant="contained"
              component={RouterLink}
              to={PATH_AUTH.loginHealthCenter}
              sx={{ mt: isSmallScreen ? 2 : 0, mr: isSmallScreen ? 0 : 2 }}
            >
              Health Center Portal
            </Button>
          </Stack>
          <LoginForm callerPage={'LoginHealthCenter'}/>
        </ContentStyle>
      </Container>
    </RootStyle>
  );
}