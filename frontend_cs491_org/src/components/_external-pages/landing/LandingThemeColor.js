import { Box, Container, Typography } from '@material-ui/core'
// material
import { alpha, styled } from '@material-ui/core/styles'
import { motion } from 'framer-motion'
// hooks
import useSettings from '../../../hooks/useSettings'
//
import { MotionInView, varFadeInDown, varFadeInUp } from '../../animate'

// ----------------------------------------------------------------------

const RootStyle = styled('div')(({ theme }) => ({
  padding: theme.spacing(15, 0),
  backgroundImage:
    theme.palette.mode === 'light'
      ? `linear-gradient(180deg, ${theme.palette.grey[300]} 0%, ${alpha(theme.palette.grey[300], 0)} 100%)`
      : 'none',
}))

// ----------------------------------------------------------------------

export default function LandingThemeColor() {
  const { themeColor, onChangeColor, colorOption } = useSettings()

  return (
    <RootStyle>
      <Container maxWidth="lg" sx={{ position: 'relative', textAlign: 'center' }}>
        <MotionInView variants={varFadeInUp}>
          <Typography component="p" variant="overline" sx={{ mb: 2, color: 'text.disabled', display: 'block' }}>
            INTUITIVE UI DESIGN
          </Typography>
        </MotionInView>

        <MotionInView variants={varFadeInUp}>
          <Typography variant="h2" sx={{ mb: 3 }}>
            Variant Analysis at Ease
          </Typography>
        </MotionInView>

        <MotionInView variants={varFadeInUp}>
          <Typography
            sx={{
              color: (theme) => (theme.palette.mode === 'light' ? 'text.secondary' : 'text.primary'),
            }}
          >
            Modern analysis software that doesn't get in the way.
          </Typography>
        </MotionInView>

        <Box sx={{ position: 'relative' }}>
          <Box component="img" src="/static/home/theme-color/grid.png" />

          <Box sx={{ position: 'absolute', top: 0 }}>
            <MotionInView variants={varFadeInUp}>
              <img
                alt="screen"
                src={`/static/home/theme-color/screen-${themeColor}.png`}
                style={{ imageRendering: 'high-quality' }}
              />
            </MotionInView>
          </Box>

          <Box sx={{ position: 'absolute', top: 0 }}>
            <MotionInView variants={varFadeInDown}>
              <motion.div
                animate={{
                  y: [-40, 0, -40],
                  scale: [1.15, 1, 1.15],
                }}
                transition={{ duration: 8, repeat: Infinity }}
              >
                <img
                  alt="sidebar"
                  src={`/static/home/theme-color/block1-${themeColor}.png`}
                  style={{ imageRendering: 'high-quality' }}
                />
              </motion.div>
            </MotionInView>
          </Box>

          <Box sx={{ position: 'absolute', top: 0 }}>
            <MotionInView variants={varFadeInDown}>
              <motion.div
                animate={{
                  y: [50, 0, 50],
                  x: [25, 0, 25],
                }}
                transition={{ duration: 8, repeat: Infinity }}
              >
                <img
                  alt="sidebar"
                  src={`/static/home/theme-color/block2-${themeColor}.png`}
                  style={{ imageRendering: 'high-quality' }}
                />
              </motion.div>
            </MotionInView>
          </Box>

          <Box sx={{ position: 'absolute', top: 0 }}>
            <MotionInView variants={varFadeInDown}>
              <motion.div animate={{ y: [-25, 5, -25] }} transition={{ duration: 10, repeat: Infinity }}>
                <img
                  alt="sidebar"
                  src={`/static/home/theme-color/sidebar-${themeColor}.png`}
                  style={{ imageRendering: 'high-quality' }}
                />
              </motion.div>
            </MotionInView>
          </Box>
        </Box>
      </Container>
    </RootStyle>
  )
}
