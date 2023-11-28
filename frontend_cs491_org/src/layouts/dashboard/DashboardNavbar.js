import menu2Fill from '@iconify/icons-eva/menu-2-fill'
import { Icon } from '@iconify/react'
import { AppBar, Box, IconButton, Stack, Toolbar } from '@material-ui/core'
// material
import { alpha, styled } from '@material-ui/core/styles'
import PropTypes from 'prop-types'
import { Link as RouterLink } from 'react-router-dom'
// components
import Logo from '../../components/Logo'
import { MHidden } from '../../components/@material-extend'
// hooks
import useCollapseDrawer from '../../hooks/useCollapseDrawer'
import AccountPopover from './AccountPopover'

// ----------------------------------------------------------------------

const DRAWER_WIDTH = 280
const COLLAPSE_WIDTH = 0 // change if sidebar is added

const APPBAR_MOBILE = 64
const APPBAR_DESKTOP = 92

const RootStyle = styled(AppBar)(({ theme }) => ({
  boxShadow: 'none',
  backdropFilter: 'blur(6px)',
  WebkitBackdropFilter: 'blur(6px)', // Fix on Mobile
  backgroundColor: alpha(theme.palette.background.default, 0.72),
  [theme.breakpoints.up('lg')]: {
    width: `calc(100% - ${DRAWER_WIDTH}px)`,
  },
}))

const ToolbarStyle = styled(Toolbar)(({ theme }) => ({
  minHeight: APPBAR_MOBILE,
  [theme.breakpoints.up('lg')]: {
    padding: theme.spacing(0, 5),
  },
}))

// ----------------------------------------------------------------------

DashboardNavbar.propTypes = {
  onOpenSidebar: PropTypes.func,
}

export default function DashboardNavbar({ onOpenSidebar }) {
  const { isCollapse } = useCollapseDrawer()

  return (
    <RootStyle
      sx={{
        ...(isCollapse && {
          width: { lg: `calc(100% - ${COLLAPSE_WIDTH}px)` },
        }),
      }}
    >
      <ToolbarStyle>
        <MHidden width="lgUp">
          <IconButton onClick={onOpenSidebar} sx={{ mr: 1, color: 'text.primary' }}>
            <Icon icon={menu2Fill} />
          </IconButton>
        </MHidden>

        <Box component={RouterLink} to="/" sx={{ display: 'inline-flex' }}>
          <Logo />
        </Box>
        <Box ml={2} />

        {/* <Searchbar /> */}
        <Box id="custom-toolbar-container" display="flex" justifyContent="center" alignItems="center" />

        <Box sx={{ flexGrow: 1 }} />

        <Stack direction="row" alignItems="center" spacing={{ xs: 0.5, sm: 1.5 }}>
          {/* <LanguagePopover /> */}
          {/* <NotificationsPopover /> */}
          {/* <ContactsPopover /> */}
          <AccountPopover />
        </Stack>
      </ToolbarStyle>
    </RootStyle>
  )
}
