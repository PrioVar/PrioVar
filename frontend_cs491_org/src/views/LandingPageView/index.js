import { makeStyles } from '@material-ui/styles'
import React from 'react'
import Page from 'src/components/Page'
import Footer from './Footer'
import Hero from './Hero'

// ----------------------------------------------------------------------

const useStyles = makeStyles((theme) => ({
  root: {
    height: '100%',
  },
  content: {
    overflow: 'hidden',
    position: 'relative',
    backgroundColor: theme.palette.background.default,
  },
}))

const LandingPageView = function () {
  const classes = useStyles()

  return (
    <Page title="Libra" id="move_top" className={classes.root}>
      <Hero />
      <div className={classes.content}>
        <Footer />
      </div>
    </Page>
  )
}

export default LandingPageView
