import arrowIosBackFill from '@iconify/icons-eva/arrow-ios-back-fill'
import { Icon } from '@iconify/react'
import { Box, Button, Container, Link, Typography } from '@material-ui/core'
// material
import { styled } from '@material-ui/core/styles'
import { Link as RouterLink } from 'react-router-dom'
import { VerifyCodeForm } from '../../components/authentication/verify-code'
// components
import Page from '../../components/Page'
// layouts
import LogoOnlyLayout from '../../layouts/LogoOnlyLayout'
// routes
import { PATH_AUTH } from '../../routes/paths'

// ----------------------------------------------------------------------

const RootStyle = styled(Page)(({ theme }) => ({
  display: 'flex',
  minHeight: '100%',
  alignItems: 'center',
  padding: theme.spacing(12, 0),
}))

// ----------------------------------------------------------------------

export default function VerifyCode() {
  return (
    <RootStyle title="Verify | Minimal UI">
      <LogoOnlyLayout />

      <Container>
        <Box sx={{ maxWidth: 480, mx: 'auto' }}>
          <Button
            size="small"
            component={RouterLink}
            to={PATH_AUTH.login}
            startIcon={<Icon icon={arrowIosBackFill} width={20} height={20} />}
            sx={{ mb: 3 }}
          >
            Back
          </Button>

          <Typography variant="h3" paragraph>
            Please check your email!
          </Typography>
          <Typography sx={{ color: 'text.secondary' }}>
            We have emailed a 6-digit confirmation code to acb@domain, please enter the code in below box to verify your
            email.
          </Typography>

          <Box sx={{ mt: 5, mb: 3 }}>
            <VerifyCodeForm />
          </Box>

          <Typography variant="body2" align="center">
            Don’t have a code? &nbsp;
            <Link variant="subtitle2" underline="none" onClick={() => {}}>
              Resend code
            </Link>
          </Typography>
        </Box>
      </Container>
    </RootStyle>
  )
}