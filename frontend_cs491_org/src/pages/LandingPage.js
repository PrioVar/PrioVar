// material
import { styled } from '@material-ui/core/styles'
import { LandingHero, LandingMinimal, LandingThemeColor } from '../components/_external-pages/landing'
// components
import Page from '../components/Page'

// ----------------------------------------------------------------------

const RootStyle = styled(Page)({
  height: '100%',
})

const ContentStyle = styled('div')(({ theme }) => ({
  overflow: 'hidden',
  position: 'relative',
  backgroundColor: theme.palette.background.default,
}))

// ----------------------------------------------------------------------

export default function LandingPage() {
  return (
    <RootStyle title="PrioVar" id="move_top">
      <LandingHero />
      <ContentStyle>
        <LandingMinimal />
        <LandingThemeColor />
      </ContentStyle>
    </RootStyle>
  )
}
