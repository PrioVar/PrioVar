// material
import { Box, Card, Container, Grid, Typography, useMediaQuery } from '@material-ui/core'
import { alpha, styled, useTheme } from '@material-ui/core/styles'
import DescriptionIcon from '@material-ui/icons/Description'
import PersonSearchIcon from '@material-ui/icons/PersonSearch'
import PublicIcon from '@material-ui/icons/Public'
import ScreenSearchDesktopIcon from '@material-ui/icons/ScreenSearchDesktop'
import LibraTM from 'src/components/LibraTM'
//
import { MotionInView, varFadeInDown, varFadeInUp } from '../../animate'

// ----------------------------------------------------------------------

const CARDS = [
  {
    icon: '/static/icons/navbar/ic_dashboard.svg',
    title: 'Load and Analyse Variants Quickly',
    description: 'PrioVar loads WGS/WES variants under 1 minute per sample and allows instant filtering analysis.',
  },
  {
    icon: <PersonSearchIcon fontSize="large" />,
    title: 'High Risk Variant Detection based on Clinical Expertise',
    description:
      'PrioVar finds high risk variants using many algorithms based on clinical expertise, eliminating the tedious manual inspection and filtering processes.',
  },
  {
    icon: <ScreenSearchDesktopIcon fontSize="large" />,
    title: 'Detailed Variant Annotation',
    description:
      "PrioVar annotates each variant and their transcripts using many pathogenicity, frequency and effect prediction databases. Hence, you don't miss risky variants.",
  },
  {
    icon: <PublicIcon fontSize="large" />,
    title: 'Population Allele Frequency Utilisation',
    description:
      'PrioVar maintains distinct allele frequency databases for each population, accounting for misleading regional polymorphisms in the diagnosis process.',
  },
  {
    icon: <DescriptionIcon fontSize="large" />,
    title: 'Diagnosis Report',
    description: 'PrioVar allows you to save any resulting variant during analysis, creating customisable reports.',
  },
]

const shadowIcon = (color) => `drop-shadow(2px 2px 2px ${alpha(color, 0.48)})`

const RootStyle = styled('div')(({ theme }) => ({
  paddingTop: theme.spacing(15),
  [theme.breakpoints.up('md')]: {
    paddingBottom: theme.spacing(15),
  },
}))

const CardStyle = styled(Card)(({ theme }) => {
  const shadowCard = (opacity) =>
    theme.palette.mode === 'light'
      ? alpha(theme.palette.grey[500], opacity)
      : alpha(theme.palette.common.black, opacity)

  return {
    maxWidth: 380,
    minHeight: 440,
    margin: 'auto',
    textAlign: 'center',
    padding: theme.spacing(10, 5, 0),
    boxShadow: `-40px 40px 80px 0 ${shadowCard(0.48)}`,
    [theme.breakpoints.up('md')]: {
      boxShadow: 'none',
      backgroundColor: theme.palette.grey[theme.palette.mode === 'light' ? 200 : 800],
    },
    '&.cardLeft': {
      [theme.breakpoints.up('md')]: { marginTop: -40 },
    },
    '&.cardCenter': {
      [theme.breakpoints.up('md')]: {
        marginTop: -80,
        backgroundColor: theme.palette.background.paper,
        boxShadow: `-40px 40px 80px 0 ${shadowCard(0.4)}`,
        '&:before': {
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          zIndex: -1,
          content: "''",
          margin: 'auto',
          position: 'absolute',
          width: 'calc(100% - 40px)',
          height: 'calc(100% - 40px)',
          borderRadius: theme.shape.borderRadiusMd,
          backgroundColor: theme.palette.background.paper,
          boxShadow: `-20px 20px 40px 0 ${shadowCard(0.12)}`,
        },
      },
    },
  }
})

const CardIconStyle = styled('img')(({ theme }) => ({
  width: 40,
  height: 40,
  margin: 'auto',
  marginBottom: theme.spacing(10),
  filter: shadowIcon(theme.palette.primary.main),
}))
const CardIconStyleDiv = styled('div')(({ theme }) => ({
  width: 40,
  height: 40,
  margin: 'auto',
  marginBottom: theme.spacing(10),
  filter: shadowIcon(theme.palette.primary.main),
}))

// ----------------------------------------------------------------------

export default function LandingMinimalHelps() {
  const theme = useTheme()
  const isLight = theme.palette.mode === 'light'
  const isDesktop = useMediaQuery(theme.breakpoints.up('lg'))

  return (
    <RootStyle>
      <Container maxWidth="lg">
        <Box sx={{ mb: { xs: 10, md: 25 } }}>
          {/* <MotionInView variants={varFadeInUp}>
            <Typography component="p" variant="overline" sx={{ mb: 2, color: 'text.secondary', textAlign: 'center' }}>
              LIBRA
            </Typography>
          </MotionInView>
          */}
          <MotionInView variants={varFadeInDown}>
            <Typography variant="h2" sx={{ textAlign: 'center' }}>
              How <LibraTM size="medium" /> helps you
            </Typography>
          </MotionInView>
        </Box>

        <Grid container spacing={isDesktop ? 10 : 5} justifyContent="center">
          {CARDS.map((card, index) => (
            <Grid key={card.title} item xs={12} md={4}>
              <MotionInView variants={varFadeInUp}>
                <CardStyle className={(index === 0 && 'cardLeft') || (index === 1 && 'cardCenter')}>
                  {typeof card.icon === 'string' ? (
                    <CardIconStyle
                      src={card.icon}
                      alt={card.title}
                      sx={{
                        ...(index % 3 === 0 && {
                          filter: (theme) => shadowIcon(theme.palette.info.main),
                        }),
                        ...(index % 3 === 1 && {
                          filter: (theme) => shadowIcon(theme.palette.error.main),
                        }),
                      }}
                    />
                  ) : (
                    <CardIconStyleDiv
                      alt={card.title}
                      sx={{
                        ...(index % 3 === 0 && {
                          filter: (theme) => shadowIcon(theme.palette.info.main),
                        }),
                        ...(index % 3 === 1 && {
                          filter: (theme) => shadowIcon(theme.palette.error.main),
                        }),
                      }}
                    >
                      {card.icon}
                    </CardIconStyleDiv>
                  )}
                  <Typography variant="h5" paragraph>
                    {card.title}
                  </Typography>
                  <Typography sx={{ color: isLight ? 'text.secondary' : 'common.white' }}>
                    {card.description}
                  </Typography>
                </CardStyle>
              </MotionInView>
            </Grid>
          ))}
        </Grid>
      </Container>
    </RootStyle>
  )
}
