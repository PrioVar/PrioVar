import { Box } from '@material-ui/core'
import { styled } from '@material-ui/core/styles'

function GroupedItems({ component = Box, ...rest }) {
  const ComponentWithStyle = styled(component)(({ theme }) => ({
    borderRadius: 8,
    border: `solid 1px ${theme.palette.grey[400_8]}`,
    backgroundColor: theme.palette.grey[400_12],
  }))

  return <ComponentWithStyle {...rest} />
}

export default GroupedItems
