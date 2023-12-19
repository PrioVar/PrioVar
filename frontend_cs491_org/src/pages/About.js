// material
import { Divider } from '@material-ui/core'
import { styled } from '@material-ui/core/styles'
import { AboutHero, AboutTeam, AboutWhat } from '../components/_external-pages/about'
// components
import Page from '../components/Page'

// ----------------------------------------------------------------------

const RootStyle = styled(Page)(({ theme }) => ({
  paddingTop: theme.spacing(8),
  [theme.breakpoints.up('md')]: {
    paddingTop: theme.spacing(11),
  },
}))

// ----------------------------------------------------------------------

export default function About() {
  return (
    <RootStyle title="About us | PrioVar">
      <AboutHero />
      <AboutWhat />
      <Divider orientation="vertical" sx={{ my: 10, mx: 'auto', width: 2, height: 40 }} />
      {/* <AboutTeam /> */}
    </RootStyle>
  )
}
