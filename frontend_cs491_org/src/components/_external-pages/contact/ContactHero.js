import { Box, Container, Typography } from '@material-ui/core'
// material
import { styled } from '@material-ui/core/styles'
import { motion } from 'framer-motion'
//
import { TextAnimate, varFadeIn, varFadeInRight, varWrapEnter } from '../../animate'

// ----------------------------------------------------------------------

const RootStyle = styled(motion.div)(({ theme }) => ({
  backgroundSize: 'cover',
  backgroundPosition: 'center',
  backgroundImage: 'url(/static/overlay.svg)',
  padding: theme.spacing(10, 0),
  [theme.breakpoints.up('md')]: {
    height: 560,
    padding: 0,
  },
}))

const ContentStyle = styled('div')(({ theme }) => ({
  textAlign: 'center',
  [theme.breakpoints.up('md')]: {
    textAlign: 'left',
    position: 'absolute',
    bottom: theme.spacing(10),
  },
}))

// ----------------------------------------------------------------------

export default function ContactHero() {
  return (
    <RootStyle initial="initial" animate="animate" variants={varWrapEnter}>
      <Container maxWidth="lg" sx={{ position: 'relative', height: '100%' }}>
        <ContentStyle>
          <TextAnimate text="Contact" sx={{ color: 'primary.main' }} variants={varFadeInRight} />
          <TextAnimate text="Us" sx={{ color: 'common.white', pl: 2 }} />
          <br />
          <Box sx={{ mt: 5, color: 'common.white' }}>
            <motion.div variants={varFadeIn}>
              <Typography variant="h6" paragraph>
                PrioVar
              </Typography>
            </motion.div>
            <motion.div variants={varFadeInRight}>
              <Typography variant="body2">
                Bilkent Cyberpark, Üniversiteler Mah. 1605. Cd. N:3/1 - B01, <br /> Çankaya, Ankara 06800 Türkiye,{' '}
                <br /> info@lidyagenomics.com
              </Typography>
            </motion.div>
          </Box>
        </ContentStyle>
      </Container>
    </RootStyle>
  )
}
