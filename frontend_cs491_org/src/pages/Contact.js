// material
import { Container, Divider, Grid } from '@material-ui/core'
import { styled } from '@material-ui/core/styles'
import { ContactEmail, ContactForm, ContactHero } from '../components/_external-pages/contact'
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

export default function Contact() {
  return (
    <RootStyle title="Contact us | PrioVar">
      <ContactHero />

      {/* <Container sx={{ my: 10 }}>
        <Grid container spacing={4}>
          <Grid item xs={12} md={7}>
            <ContactForm />
          </Grid>

          <Grid container item md={1} justifyContent="center">
            <Divider orientation="vertical" />
          </Grid>

          <Grid item xs={12} md={4}>
            <ContactEmail />
          </Grid>
        </Grid>
      </Container> */}
    </RootStyle>
  )
}
