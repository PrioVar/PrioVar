import instagramFilled from '@iconify/icons-ant-design/instagram-filled'
import facebookFill from '@iconify/icons-eva/facebook-fill'
import linkedinFill from '@iconify/icons-eva/linkedin-fill'
import twitterFill from '@iconify/icons-eva/twitter-fill'
import { Icon } from '@iconify/react'
import { Box, Button, Container, Typography } from '@material-ui/core'
// material
import { styled } from '@material-ui/core/styles'
import { Link as RouterLink } from 'react-router-dom'
import { ComingSoonIllustration } from '../assets'
// components
import Page from '../components/Page'
// hooks
import useCountdown from '../hooks/useCountdown'

// ----------------------------------------------------------------------

const SOCIALS = [
  {
    name: 'Facebook',
    icon: <Icon icon={facebookFill} width={24} height={24} color="#1877F2" />,
  },
  {
    name: 'Instagram',
    icon: <Icon icon={instagramFilled} width={24} height={24} color="#D7336D" />,
  },
  {
    name: 'Linkedin',
    icon: <Icon icon={linkedinFill} width={24} height={24} color="#006097" />,
  },
  {
    name: 'Twitter',
    icon: <Icon icon={twitterFill} width={24} height={24} color="#1C9CEA" />,
  },
]

const RootStyle = styled(Page)(({ theme }) => ({
  minHeight: '100%',
  display: 'flex',
  alignItems: 'center',
  paddingTop: theme.spacing(15),
  paddingBottom: theme.spacing(10),
}))

const CountdownStyle = styled('div')(({ theme }) => ({
  display: 'flex',
  justifyContent: 'center',
  paddingBottom: theme.spacing(5),
}))

const SeparatorStyle = styled(Typography)(({ theme }) => ({
  margin: theme.spacing(0, 1),
  [theme.breakpoints.up('sm')]: {
    margin: theme.spacing(0, 2.5),
  },
}))

// ----------------------------------------------------------------------

export default function ComingSoon() {
  const countdown = useCountdown(new Date('01 July 2023 14:30 UTC'))

  return (
    <RootStyle title="Coming Soon | PrioVar">
      <Container>
        <Box sx={{ maxWidth: 480, margin: 'auto', textAlign: 'center' }}>
          <Typography variant="h3" paragraph>
            Coming Soon!
          </Typography>
          <Typography sx={{ color: 'text.secondary' }}>We are working hard on this page!</Typography>

          <ComingSoonIllustration sx={{ my: 10, height: 240 }} />

          <CountdownStyle>
            <div>
              <Typography variant="h2">{countdown.days}</Typography>
              <Typography sx={{ color: 'text.secondary' }}>Days</Typography>
            </div>

            <SeparatorStyle variant="h2">:</SeparatorStyle>

            <div>
              <Typography variant="h2">{countdown.hours}</Typography>
              <Typography sx={{ color: 'text.secondary' }}>Hours</Typography>
            </div>

            <SeparatorStyle variant="h2">:</SeparatorStyle>

            <div>
              <Typography variant="h2">{countdown.minutes}</Typography>
              <Typography sx={{ color: 'text.secondary' }}>Minutes</Typography>
            </div>

            <SeparatorStyle variant="h2">:</SeparatorStyle>

            <div>
              <Typography variant="h2">{countdown.seconds}</Typography>
              <Typography sx={{ color: 'text.secondary' }}>Seconds</Typography>
            </div>
          </CountdownStyle>

          <Button variant="contained" size="large" component={RouterLink} to="/">
            Go to Home
          </Button>

          {/*
          <OutlinedInput
            fullWidth
            placeholder="Enter your email"
            endAdornment={
              <InputAdornment position="end">
                <Button variant="contained" size="large">
                  Notify Me
                </Button>
              </InputAdornment>
            }
            sx={{
              my: 5,
              pr: 0.5,
              transition: (theme) =>
                theme.transitions.create('box-shadow', {
                  easing: theme.transitions.easing.easeInOut,
                  duration: theme.transitions.duration.shorter,
                }),
              '&.Mui-focused': {
                boxShadow: (theme) => theme.customShadows.z8,
              },
              '& fieldset': {
                borderWidth: `1px !important`,
                borderColor: (theme) => `${theme.palette.grey[500_32]} !important`,
              },
            }}
          />

          <Box sx={{ textAlign: 'center', '& > *': { mx: 1 } }}>
            {SOCIALS.map((social) => (
              <Tooltip key={social.name} title={social.name}>
                <MIconButton>{social.icon}</MIconButton>
              </Tooltip>
            ))}
          </Box>
          */}
        </Box>
      </Container>
    </RootStyle>
  )
}
