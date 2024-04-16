import flashFill from '@iconify-icons/eva/flash-fill'
import { Icon } from '@iconify/react'
import { Box, Button, Container, Typography } from '@material-ui/core'
import { makeStyles } from '@material-ui/styles'
import clsx from 'clsx'
import { motion } from 'framer-motion'
import PropTypes from 'prop-types'
import React from 'react'
import { Link as RouterLink } from 'react-router-dom'
import { varFadeIn, varFadeInRight, varFadeInUp, varWrapEnter } from 'src/components/Animate'
import { PATH_APP } from 'src/routes/paths'
import { BASE_IMG } from 'src/utils/getImages'

// ----------------------------------------------------------------------

const useStyles = makeStyles((theme) => ({
  root: {
    position: 'relative',
    backgroundColor: '#F2F3F5',
    [theme.breakpoints.up('md')]: {
      top: 0,
      left: 0,
      width: '100%',
      height: '100vh',
      display: 'flex',
      position: 'fixed',
      alignItems: 'center',
    },
  },
  content: {
    zIndex: 10,
    maxWidth: 520,
    margin: 'auto',
    textAlign: 'center',
    position: 'relative',
    paddingTop: theme.spacing(15),
    paddingBottom: theme.spacing(15),
    [theme.breakpoints.up('md')]: {
      margin: 'unset',
      textAlign: 'left',
    },
  },
  heroOverlay: {
    zIndex: 9,
    width: '100%',
    height: '100%',
    objectFit: 'cover',
    position: 'absolute',
  },
  heroImg: {
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
      height: '72vh',
    },
  },
  link: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: theme.spacing(5),
    color: theme.palette.common.white,
    [theme.breakpoints.up('md')]: {
      justifyContent: 'flex-start',
    },
  },
  listIcon: {
    display: 'flex',
    marginTop: theme.spacing(5),
    justifyContent: 'center',
    [theme.breakpoints.up('md')]: {
      justifyContent: 'flex-start',
    },
    '& > :not(:last-of-type)': {
      marginRight: theme.spacing(1.5),
    },
  },
}))

// ----------------------------------------------------------------------

const getImg = (width) => `${BASE_IMG}w_${width}/v1611472901/upload_minimal/home/hero.png`

Hero.propTypes = {
  className: PropTypes.string,
}

var Hero = function ({ className }) {
  const classes = useStyles()

  return (
    <>
      <motion.div initial="initial" animate="animate" variants={varWrapEnter} className={clsx(classes.root, className)}>
        <motion.img
          alt="overlay"
          src="/static/images/overlay.svg"
          variants={varFadeIn}
          className={classes.heroOverlay}
        />

        <motion.img alt="hero" src={getImg(1200)} variants={varFadeInUp} className={classes.heroImg} />

        <Container maxWidth="lg">
          <div className={classes.content}>
            <motion.div variants={varFadeInRight}>
              <Typography variant="h1" sx={{ color: 'common.white' }}>
                Empowering Genomic Diagnosis with
                <Typography component="span" variant="h1" sx={{ color: 'primary.main' }}>
                  &nbsp;Libra™
                </Typography>
              </Typography>
            </motion.div>

            <motion.div variants={varFadeInRight}>
              <Box component="p" sx={{ color: 'common.white', py: 5 }}>
                Genomic variant analysis software to provide clinically meaningful answers, not just variants.
              </Box>
            </motion.div>

            <motion.div variants={varFadeInRight}>
              <Button
                size="large"
                variant="contained"
                component={RouterLink}
                to={PATH_APP.app.fileUpload}
                startIcon={<Icon icon={flashFill} width={20} height={20} />}
              >
                Free Trial
              </Button>
            </motion.div>
          </div>
        </Container>
      </motion.div>
      <Box sx={{ height: { md: '100vh' } }} />
    </>
  )
}

export default Hero
