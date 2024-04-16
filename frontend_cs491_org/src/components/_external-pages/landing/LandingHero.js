import flashFill from '@iconify/icons-eva/flash-fill'
import { Icon } from '@iconify/react'
import { Box, Button, Container, Stack, Typography, useMediaQuery, useTheme } from '@material-ui/core'
// material
import { styled } from '@material-ui/core/styles'
import { motion } from 'framer-motion'
import { Link as RouterLink } from 'react-router-dom'
import LibraTM from 'src/components/LibraTM'
// routes
import { PATH_DASHBOARD } from '../../../routes/paths'
//
import { varFadeIn, varFadeInRight, varFadeInUp, varWrapEnter } from '../../animate'

// ----------------------------------------------------------------------

const RootStyle = styled(motion.div)(({ theme }) => ({
  position: 'relative',
  backgroundColor: theme.palette.grey[400],
  [theme.breakpoints.up('md')]: {
    top: 0,
    left: 0,
    width: '100%',
    height: '100vh',
    display: 'flex',
    position: 'fixed',
    alignItems: 'center',
  },
}))

const ContentStyle = styled((props) => <Stack spacing={5} {...props} />)(({ theme }) => ({
  zIndex: 10,
  maxWidth: 585,
  margin: 'auto',
  textAlign: 'center',
  position: 'relative',
  paddingTop: theme.spacing(15),
  paddingBottom: theme.spacing(15),
  [theme.breakpoints.up('md')]: {
    margin: 'unset',
    textAlign: 'left',
  },
}))

const HeroOverlayStyle = styled(motion.img)({
  zIndex: 9,
  width: '100%',
  height: '100%',
  objectFit: 'cover',
  position: 'absolute',
})

const HeroImgStyle = styled(motion.img)(({ theme }) => ({
  top: 0,
  right: 0,
  bottom: 0,
  zIndex: 8,
  width: '100%',
  margin: 'auto',
  position: 'absolute',
  [theme.breakpoints.up('lg')]: {
    right: '8%',
    width: 'auto',
    height: '48vh',
  },
}))

// ----------------------------------------------------------------------

export default function LandingHero() {
  const theme = useTheme()
  const isLight = theme.palette.mode === 'light'
  const isDesktop = useMediaQuery(theme.breakpoints.up('lg'))

  return (
    <>
      <RootStyle initial="initial" animate="animate" variants={varWrapEnter}>
        <HeroOverlayStyle
          alt="overlay"
          src="/static/overlay.svg"
          animate={{ opacity: isDesktop ? 0.09 : 0.4 }}
          transition={varFadeIn.animate.transition}
          initial={{ opacity: 0 }}
        />
        <HeroImgStyle alt="hero" src="/static/home/hero.png" variants={varFadeInUp} />

        <Container maxWidth="lg">
          <ContentStyle>
            <motion.div variants={varFadeInRight}>
              <Typography variant="h1" sx={{ color: 'common.white' }}>
                Empowering Genomic Diagnosis with&nbsp;
                <LibraTM />
              </Typography>
            </motion.div>

            <motion.div variants={varFadeInRight}>
              <Typography variant="h6" paragraph sx={{ color: 'common.white', fontWeight: 300 }}>
                The genomic variant analysis software that is designed to discover pathogenic variants quickly and
                accurately.
              </Typography>
            </motion.div>

            <motion.div variants={varFadeInRight}>
              <Button
                size="large"
                variant="contained"
                component={RouterLink}
                to={PATH_DASHBOARD.root}
                startIcon={<Icon icon={flashFill} width={20} height={20} />}
              >
                Demo
              </Button>
            </motion.div>
          </ContentStyle>
        </Container>
      </RootStyle>
      <Box sx={{ height: { md: '100vh' } }} />
    </>
  )
}
