import { Chip } from '@material-ui/core'
import React from 'react'

const ExternalLink = function (props) {
  return <a target="_blank" rel="noopener" {...props} />
}

export var ChipLink = function (props) {
  return <Chip clickable component={ExternalLink} {...props} />
}
