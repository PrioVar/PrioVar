import { Toolbar } from '@material-ui/core'
import { makeStyles } from '@material-ui/styles'
import PropTypes from 'prop-types'
import React from 'react'

const useStyles = makeStyles((theme) => ({
  root: {
    padding: 0,
  },
}))

function LibraTableToolbar({ children }) {
  const classes = useStyles()

  return <Toolbar className={classes.root}>{children}</Toolbar>
}

LibraTableToolbar.propTypes = {
  children: PropTypes.node.isRequired,
}

export default LibraTableToolbar
