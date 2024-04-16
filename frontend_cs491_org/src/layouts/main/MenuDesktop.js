import arrowIosDownwardFill from '@iconify/icons-eva/arrow-ios-downward-fill'
import arrowIosUpwardFill from '@iconify/icons-eva/arrow-ios-upward-fill'
import { Icon } from '@iconify/react'
import { Box, CardActionArea, Link, List, ListItem, Popover, Stack } from '@material-ui/core'
// material
import { styled } from '@material-ui/core/styles'
import { motion } from 'framer-motion'
import PropTypes from 'prop-types'
import { useEffect, useRef, useState } from 'react'
import { NavLink as RouterLink, useLocation } from 'react-router-dom'

// ----------------------------------------------------------------------

const LinkStyle = styled(Link)(({ theme }) => ({
  ...theme.typography.subtitle2,
  color: theme.palette.text.primary,
  marginRight: theme.spacing(5),
  transition: theme.transitions.create('opacity', {
    duration: theme.transitions.duration.shortest,
  }),
  '&:hover': {
    opacity: 0.48,
    textDecoration: 'none',
  },
}))

// ----------------------------------------------------------------------

IconBullet.propTypes = {
  type: PropTypes.oneOf(['subheader', 'item']),
}

function IconBullet({ type = 'item' }) {
  return (
    <Box sx={{ width: 24, height: 16, display: 'flex', alignItems: 'center' }}>
      <Box
        component="span"
        sx={{
          ml: '2px',
          width: 4,
          height: 4,
          borderRadius: '50%',
          bgcolor: 'currentColor',
          ...(type !== 'item' && { ml: 0, width: 8, height: 2, borderRadius: 2 }),
        }}
      />
    </Box>
  )
}

MenuDesktopItem.propTypes = {
  item: PropTypes.object,
  pathname: PropTypes.string,
  isHome: PropTypes.bool,
  isOffset: PropTypes.bool,
  isOpen: PropTypes.bool,
  onOpen: PropTypes.func,
  onClose: PropTypes.func,
}

function MenuDesktopItem({ item, pathname, isHome, isOpen, isOffset, onOpen, onClose }) {
  const linkEl = useRef(null)
  const { title, path, children } = item
  const isActive = pathname === path

  if (children) {
    return (
      <div key={title} ref={linkEl}>
        <LinkStyle
          onClick={onOpen}
          sx={{
            display: 'flex',
            cursor: 'pointer',
            alignItems: 'center',
            ...(isHome && { color: 'common.white' }),
            ...(isOffset && { color: 'text.primary' }),
            ...(isOpen && { opacity: 0.48 }),
          }}
        >
          {title}
          <Box
            component={Icon}
            icon={isOpen ? arrowIosUpwardFill : arrowIosDownwardFill}
            sx={{ ml: 0.5, width: 16, height: 16 }}
          />
        </LinkStyle>

        <Popover
          open={isOpen}
          onClose={onClose}
          anchorEl={linkEl.current}
          anchorOrigin={{
            vertical: 'bottom',
            horizontal: 'left',
          }}
          transformOrigin={{
            vertical: 'top',
            horizontal: 'left',
          }}
          PaperProps={{
            sx: {
              px: 3,
              pb: 3,
              borderRadius: 2,
              boxShadow: (theme) => theme.customShadows.z24,
            },
          }}
        >
          <List disablePadding>
            {children.map((item) => {
              return (
                <ListItem
                  key={item.title}
                  to={item.path}
                  component={RouterLink}
                  underline="none"
                  sx={{
                    p: 0,
                    mt: 3,
                    typography: 'body2',
                    color: 'text.secondary',
                    transition: (theme) => theme.transitions.create('color'),
                    '&:hover': { color: 'text.primary' },
                    ...(item.path === pathname && {
                      typography: 'subtitle2',
                      color: 'text.primary',
                    }),
                  }}
                >
                  {item.title === 'Dashboard' ? (
                    <CardActionArea
                      sx={{
                        py: 5,
                        px: 10,
                        borderRadius: 2,
                        color: 'primary.main',
                        bgcolor: 'background.neutral',
                      }}
                    >
                      <Box
                        component={motion.img}
                        whileTap="tap"
                        whileHover="hover"
                        variants={{
                          hover: { scale: 1.02 },
                          tap: { scale: 0.98 },
                        }}
                        src="/static/illustrations/illustration_dashboard.png"
                      />
                    </CardActionArea>
                  ) : (
                    <>
                      <IconBullet />
                      {item.title}
                    </>
                  )}
                </ListItem>
              )
            })}
          </List>
        </Popover>
      </div>
    )
  }

  if (title === 'Documentation') {
    return (
      <LinkStyle
        href={path}
        target="_blank"
        sx={{
          ...(isHome && { color: 'common.white' }),
          ...(isOffset && { color: 'text.primary' }),
          ...(isActive && { color: 'primary.main' }),
        }}
      >
        {title}
      </LinkStyle>
    )
  }

  return (
    <LinkStyle
      to={path}
      component={RouterLink}
      sx={{
        ...(isHome && { color: 'common.white' }),
        ...(isOffset && { color: 'text.primary' }),
        ...(isActive && { color: 'primary.main' }),
      }}
    >
      {title}
    </LinkStyle>
  )
}

MenuDesktop.propTypes = {
  isOffset: PropTypes.bool,
  isHome: PropTypes.bool,
  navConfig: PropTypes.array,
}

export default function MenuDesktop({ isOffset, isHome, navConfig }) {
  const { pathname } = useLocation()
  const [open, setOpen] = useState(false)

  useEffect(() => {
    if (open) {
      handleClose()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathname])

  const handleOpen = () => {
    setOpen(true)
  }

  const handleClose = () => {
    setOpen(false)
  }

  return (
    <Stack direction="row">
      {navConfig.map((link) => (
        <MenuDesktopItem
          key={link.title}
          item={link}
          pathname={pathname}
          isOpen={open}
          onOpen={handleOpen}
          onClose={handleClose}
          isOffset={isOffset}
          isHome={isHome}
        />
      ))}
    </Stack>
  )
}
