import clsx from 'clsx'
import React from 'react'
import PropTypes from 'prop-types'
import Logo from 'src/components/Logo'
import { Link as ScrollLink } from 'react-scroll'
import { makeStyles } from '@material-ui/styles'
import { Link, Container, Typography } from '@material-ui/core'

// ----------------------------------------------------------------------

const useStyles = makeStyles((theme) => ({
  root: {
    textAlign: 'center',
    padding: theme.spacing(5, 0),
  },
}))

// ----------------------------------------------------------------------

Footer.propTypes = {
  className: PropTypes.string,
}

var Footer = function ({ className }) {
  const classes = useStyles()

  return (
    <Container maxWidth="lg" className={clsx(classes.root, className)}>
      <ScrollLink to="move_top" spy smooth>
        <Logo sx={{ mb: 1, mx: 'auto' }} />
      </ScrollLink>

      <Typography variant="caption">
        © All rights reserved
        <br /> Made by &nbsp; PrioVar
      </Typography>
    </Container>
  )
}

export default Footer
