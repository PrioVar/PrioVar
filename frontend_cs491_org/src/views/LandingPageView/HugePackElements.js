import { Box, Button, Container, Grid, Typography } from '@material-ui/core'
import { alpha } from '@material-ui/core/styles'
import { makeStyles, useTheme } from '@material-ui/styles'
import clsx from 'clsx'
import PropTypes from 'prop-types'
import React from 'react'
import { MotionInView, varFadeInRight, varFadeInUp } from 'src/components/Animate'
import useBreakpoints from 'src/hooks/useBreakpoints'
import { BASE_IMG } from 'src/utils/getImages'

// ----------------------------------------------------------------------

const useStyles = makeStyles((theme) => {
  const isRTL = theme.direction === 'rtl'

  return {
    root: {
      padding: theme.spacing(15, 0),
      backgroundImage:
        theme.palette.mode === 'light'
          ? `linear-gradient(180deg, ${alpha(theme.palette.grey[300], 0)} 0%, ${theme.palette.grey[300]} 100%)`
          : 'none',
    },
    content: {
      maxWidth: 520,
      margin: 'auto',
      textAlign: 'center',
      marginBottom: theme.spacing(10),
      [theme.breakpoints.up('md')]: {
        height: '100%',
        marginBottom: 0,
        textAlign: 'left',
        display: 'inline-flex',
        flexDirection: 'column',
        alignItems: 'flex-start',
        justifyContent: 'center',
        paddingRight: theme.spacing(5),
      },
    },
    screen: {
      bottom: 0,
      maxWidth: 460,
      position: 'absolute',
    },
    screenLeft: { zIndex: 3 },
    screenRight: { zIndex: 1 },
    screenCenter: {
      position: 'relative',
      zIndex: 2,
      bottom: 20,
      transform: isRTL ? 'translateX(-24%)' : 'translateX(24%)',
      [theme.breakpoints.up('sm')]: {
        bottom: 40,
        transform: isRTL ? 'translateX(-32%)' : 'translateX(32%)',
      },
    },
  }
})

const variantScreenLeftMoblie = {
  initial: { x: '22%', y: -10, opacity: 0 },
  animate: { x: 0, y: 0, opacity: 1 },
}
const variantScreenRightMobile = {
  initial: { x: '26%', y: -30, opacity: 0 },
  animate: { x: '48%', y: -40, opacity: 1 },
}
const variantScreenLeft = {
  initial: { x: '30%', y: -30, opacity: 0 },
  animate: { x: 0, y: 0, opacity: 1 },
}
const variantScreenCenter = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
}
const variantScreenRight = {
  initial: { x: '34%', y: -50, opacity: 0 },
  animate: { x: '64%', y: -80, opacity: 1 },
}
const transition = { duration: 0.5, ease: 'easeOut' }

// ----------------------------------------------------------------------

HugePackElements.propTypes = {
  className: PropTypes.string,
}

var HugePackElements = function ({ className }) {
  const classes = useStyles()
  const theme = useTheme()
  const upSm = useBreakpoints('up', 'sm')
  const upMd = useBreakpoints('up', 'md')
  const textAnimate = upMd ? varFadeInRight : varFadeInUp
  const screenLeftAnimate = upSm ? variantScreenLeft : variantScreenLeftMoblie
  const screenCenterAnimate = variantScreenCenter
  const screenRightAnimate = upSm ? variantScreenRight : variantScreenRightMobile

  const getImg = (width, index) =>
    `${BASE_IMG}w_${width}/v1611472901/upload_minimal/home/screen_${
      theme.palette.mode === 'light' ? 'light' : 'dark'
    }_${index + 1}.png`

  return (
    <div className={clsx(classes.root, className)}>
      <Container maxWidth="lg">
        <Grid container spacing={5}>
          <Grid item xs={12} md={4} lg={5}>
            <div className={classes.content}>
              <MotionInView variants={textAnimate}>
                <Typography gutterBottom variant="overline" sx={{ color: 'text.secondary', display: 'block' }}>
                  Interface Starter Kit
                </Typography>
              </MotionInView>

              <MotionInView variants={textAnimate}>
                <Typography variant="h2" paragraph>
                  Huge Pack of Elements
                </Typography>
              </MotionInView>

              <MotionInView variants={textAnimate}>
                <Typography sx={{ color: 'text.secondary' }}>
                  We collected most popular elements. Menu, sliders, buttons, inputs etc. are all here. Just dive in!
                </Typography>
              </MotionInView>

              <MotionInView variants={textAnimate} sx={{ mt: 5 }}>
                <Button size="large" color="inherit" variant="outlined">
                  View All Components
                </Button>
              </MotionInView>
            </div>
          </Grid>

          <Grid
            dir="ltr"
            item
            xs={12}
            md={8}
            lg={7}
            sx={{
              position: 'relative',
              pl: { sm: '16% !important', md: '0 !important' },
            }}
          >
            {[...Array(3)].map((screen, index) => (
              <MotionInView
                key={index}
                variants={
                  (index === 0 && screenLeftAnimate) || (index === 1 && screenCenterAnimate) || screenRightAnimate
                }
                transition={transition}
                className={clsx(classes.screen, {
                  [classes.screenLeft]: index === 0,
                  [classes.screenCenter]: index === 1,
                  [classes.screenRight]: index === 2,
                })}
              >
                <Box
                  component="img"
                  alt={`screen ${index + 1}`}
                  src={getImg(720, index)}
                  variants={varFadeInUp}
                  className="lazyload"
                  sx={{ width: { xs: '80%', sm: '100%' } }}
                />
              </MotionInView>
            ))}
          </Grid>
        </Grid>
      </Container>
    </div>
  )
}

export default HugePackElements
