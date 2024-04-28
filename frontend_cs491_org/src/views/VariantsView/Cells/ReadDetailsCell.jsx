import { Stack, Tooltip, Typography } from '@material-ui/core'
import PropTypes from 'prop-types'
import React from 'react'
import Label from 'src/components/Label'
import { toFixedTrunc } from 'src/utils/math'

const renderChip = (title, value, tooltip = '') => {
  return (
    <Tooltip title={tooltip}>
      <Stack direction="column" spacing={0.25}>
        <Typography variant="caption" color="text.secondary" sx={{ fontSize: 11, textAlign: 'center' }}>
          {title}
        </Typography>
        <Label variant="ghost">{value}</Label>
      </Stack>
    </Tooltip>
  )
}

const ReadDetailsCell = function ({ dp, adList }) {
  return (
    <Stack direction="row" spacing={0.5}>
      {renderChip('DP', dp, 'Read Depth')}
      {renderChip('AD', adList.join(' | '), 'Allelic Depth')}
      {renderChip(
        'AB',
        adList.slice(1).map((ad) => toFixedTrunc(dp / ad, 2)),
        'Allelic Balance',
      )}
    </Stack>
  )
}

ReadDetailsCell.propTypes = {
  dp: PropTypes.number.isRequired,
  adList: PropTypes.arrayOf(PropTypes.number).isRequired,
}

export default ReadDetailsCell
