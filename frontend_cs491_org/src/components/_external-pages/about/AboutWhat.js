import PropTypes from 'prop-types'
import { Icon } from '@iconify/react'
import roundArrowRightAlt from '@iconify/icons-ic/round-arrow-right-alt'
// material
import { alpha, useTheme, styled } from '@material-ui/core/styles'
import { Box, Grid, Button, Container, Typography, LinearProgress } from '@material-ui/core'
// utils
import { fPercent } from '../../../utils/formatNumber'
import mockData from '../../../utils/mock-data'
//
import { MHidden } from '../../@material-extend'
import { varFadeInUp, varFadeInRight, MotionInView } from '../../animate'

// ----------------------------------------------------------------------

const LABEL = ['Development', 'Design', 'Marketing']

const MOCK_SKILLS = [...Array(3)].map((_, index) => ({
  label: LABEL[index],
  value: mockData.number.percent(index),
}))

const RootStyle = styled('div')(({ theme }) => ({
  textAlign: 'center',
  paddingTop: theme.spacing(10),
  paddingBottom: theme.spacing(1),
  [theme.breakpoints.up('md')]: {
    textAlign: 'left',
  },
}))

// ----------------------------------------------------------------------

ProgressItem.propTypes = {
  progress: PropTypes.shape({
    label: PropTypes.string,
    value: PropTypes.number,
  }),
}

function ProgressItem({ progress }) {
  const { label, value } = progress
  return (
    <Box sx={{ mt: 3 }}>
      <Box sx={{ mb: 1.5, display: 'flex', alignItems: 'center' }}>
        <Typography variant="subtitle2">{label}&nbsp;-&nbsp;</Typography>
        <Typography variant="body2" sx={{ color: 'text.secondary' }}>
          {fPercent(value)}
        </Typography>
      </Box>
      <LinearProgress
        variant="determinate"
        value={value}
        sx={{
          '& .MuiLinearProgress-bar': { bgcolor: 'grey.700' },
          '&.MuiLinearProgress-determinate': { bgcolor: 'divider' },
        }}
      />
    </Box>
  )
}

export default function AboutWhat() {
  const theme = useTheme()
  const isLight = theme.palette.mode === 'light'
  const shadow = `-40px 40px 80px ${alpha(isLight ? theme.palette.grey[500] : theme.palette.common.black, 0.48)}`

  return (
    <RootStyle>
      <Container maxWidth="lg">
        <Grid container spacing={3}>
          {/* <MHidden width="mdDown"> */}
          {/* <Grid item xs={12} md={6} lg={7} sx={{ pr: { md: 7 } }}> */}
          {/* <Grid container spacing={3} alignItems="flex-end">
                <Grid item xs={6}>
                  <MotionInView variants={varFadeInUp}>
                    <Box
                      component="img"
                      src="/static/about/what-1.jpg"
                      sx={{
                        borderRadius: 2,
                        boxShadow: shadow,
                      }}
                    />
                  </MotionInView>
                </Grid>
                <Grid item xs={6}>
                  <MotionInView variants={varFadeInUp}>
                    <Box component="img" src="/static/about/what-2.jpg" sx={{ borderRadius: 2 }} />
                  </MotionInView>
                </Grid>
              </Grid> */}
          {/* </Grid> */}
          {/* </MHidden> */}

          <Grid item xs={12} md={6} lg={14}>
            <MotionInView variants={varFadeInRight}>
              <Typography variant="h2" sx={{ mb: 1 }}>
                About PrioVar
              </Typography>
            </MotionInView>

            <MotionInView variants={varFadeInRight}>
              <Typography
                sx={{
                  color: (theme) => (theme.palette.mode === 'light' ? 'text.secondary' : 'common.white'),
                }}
              >
                At PrioVar, we are passionate about unraveling the mysteries of the human genome and transforming
                the way we understand and approach healthcare. As a leading provider of innovative solutions for genomic
                data analysis, we are dedicated to empowering researchers, scientists, and healthcare professionals with
                the tools and knowledge they need to make discoveries and improve patient outcomes.
                <br />
                <br />
                Founded in 2020, PrioVar was born out of a vision to bridge the gap between cutting-edge genomic
                technologies and their practical application in various domains. We recognized the immense potential of
                genomics to revolutionize personalized medicine, and scientific research, and we set out to create a
                company that could harness this potential to drive meaningful progress in the field.
                <br />
                <br />
                Our team at PrioVar comprises a diverse group of experts from various disciplines, including
                genomics, bioinformatics, and computer science. This multidisciplinary approach enables us to develop
                comprehensive solutions that integrate the latest advancements in technology with the insights gained
                from genomic data analysis.
                <br />
                <br />
                We offer a range of powerful and user-friendly software tools and platforms tailored to meet the
                specific needs of researchers and healthcare professionals. Our solutions leverage state-of-the-art
                algorithms, machine learning techniques, and data visualization capabilities to provide accurate,
                actionable, and interpretable results. Whether you are studying the genetic basis of diseases,
                investigating the effectiveness of targeted therapies, our tools empower you to extract meaningful
                insights from complex genomic data quickly and efficiently.
                <br />
                <br />
                At PrioVar, we are committed to driving scientific discovery. We believe that by democratizing
                access to advanced genomic analysis tools, we can foster collaboration, accelerate research, and enable
                the development of personalized treatments for a wide range of diseases. Our goal is to empower our
                customers to push the boundaries of genomics and unlock the full potential of this transformative field.
                <br />
                <br />
                As we continue to expand our offerings and explore new avenues in genomics, we remain steadfast in our
                commitment to excellence, innovation, and customer satisfaction. We strive to provide exceptional
                support and guidance to our clients, ensuring that they maximize the value of our solutions and achieve
                their research and clinical objectives.
                <br />
                <br />
                Join us on our journey as we revolutionize genomic analysis and pave the way for a future where
                precision medicine is accessible to all. Together, let's unlock the secrets of the genome and transform
                healthcare for generations to come.
              </Typography>
            </MotionInView>
          </Grid>
        </Grid>
      </Container>
    </RootStyle>
  )
}
