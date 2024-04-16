import { AppBar, Box, Button, Container, Toolbar } from '@material-ui/core'
// material
import { styled } from '@material-ui/core/styles'
import { NavLink as RouterLink, useLocation } from 'react-router-dom'
import { PATH_PAGE } from 'src/routes/paths'
import { MHidden } from '../../components/@material-extend'
// components
import Logo from '../../components/Logo'
// hooks
import useOffSetTop from '../../hooks/useOffSetTop'
import navConfig from './MenuConfig'
//
import MenuDesktop from './MenuDesktop'
import MenuMobile from './MenuMobile'

// ----------------------------------------------------------------------

const APP_BAR_MOBILE = 64
const APP_BAR_DESKTOP = 88

const ToolbarStyle = styled(Toolbar)(({ theme }) => ({
  height: APP_BAR_MOBILE,
  transition: theme.transitions.create(['height', 'background-color'], {
    easing: theme.transitions.easing.easeInOut,
    duration: theme.transitions.duration.shorter,
  }),
  [theme.breakpoints.up('md')]: {
    height: APP_BAR_DESKTOP,
  },
}))

const ToolbarShadowStyle = styled('div')(({ theme }) => ({
  left: 0,
  right: 0,
  bottom: 0,
  height: 24,
  zIndex: -1,
  margin: 'auto',
  borderRadius: '50%',
  position: 'absolute',
  width: `calc(100% - 48px)`,
  boxShadow: theme.customShadows.z8,
}))

// ----------------------------------------------------------------------

export default function MainNavbar() {
  const isOffset = useOffSetTop(100)
  const { pathname } = useLocation()
  const isHome = pathname === '/'

  return (
    <AppBar sx={{ boxShadow: 0, bgcolor: 'transparent' }}>
      <ToolbarStyle
        disableGutters
        sx={{
          ...(isOffset && {
            bgcolor: 'background.default',
            height: { md: APP_BAR_DESKTOP - 16 },
          }),
        }}
      >
        <Container
          maxWidth="lg"
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <RouterLink to="/">
            <Logo />
          </RouterLink>

          <Box sx={{ flexGrow: 1 }} />

          <MHidden width="mdDown">
            <MenuDesktop isOffset={isOffset} isHome={isHome} navConfig={navConfig} />
          </MHidden>


          <MHidden width="mdUp">
            <MenuMobile isOffset={isOffset} isHome={isHome} navConfig={navConfig} />
          </MHidden>
        </Container>
      </ToolbarStyle>

      {isOffset && <ToolbarShadowStyle />}
    </AppBar>
  )
}
