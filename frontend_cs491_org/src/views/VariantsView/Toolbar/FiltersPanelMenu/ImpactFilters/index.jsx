import { Stack } from '@material-ui/core'

import ImpactChoices from './ImpactChoices'

function ImpactFilters(props) {
  return (
    <Stack direction="column">
      <ImpactChoices
        impactEvent={props.impactEvent}
        setImpactEvent={props.setImpactEvent}
        filterType={props.filterType}
      />
    </Stack>
  )
}

export default ImpactFilters
