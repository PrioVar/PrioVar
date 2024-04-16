// material
import { Container } from '@material-ui/core'
import { styled } from '@material-ui/core/styles'
import {
  ComponentFoundation,
  ComponentHero,
  ComponentMaterialUI,
  ComponentOther,
} from '../components/_external-pages/components-overview'
// components
import Page from '../components/Page'

// ----------------------------------------------------------------------

const RootStyle = styled(Page)(({ theme }) => ({
  paddingTop: theme.spacing(8),
  paddingBottom: theme.spacing(15),
  [theme.breakpoints.up('md')]: {
    paddingTop: theme.spacing(11),
  },
}))

// ----------------------------------------------------------------------

export default function ComponentsOverview() {
  return (
    <RootStyle title="Components Overview | PrioVar">
      <ComponentHero />
      <Container maxWidth="lg">
        <ComponentFoundation />
        <ComponentMaterialUI />
        <ComponentOther />
      </Container>
    </RootStyle>
  )
}
