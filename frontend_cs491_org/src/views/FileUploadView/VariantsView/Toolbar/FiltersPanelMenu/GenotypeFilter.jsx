import { chain as flatMap } from 'ramda'
import { Stack } from '@material-ui/core'
import Tags from 'src/components/Tags'
import YesNoAnyButtonGroup from 'src/views/VariantsView/Toolbar/FiltersPanelMenu/YesNoAnyButtonGroup'

const toTagOption = (value) => {
  switch (value) {
    case 'Heterozygote':
    case 'Homozygote':
      return { label: value, value, hidden: true }
    default:
      return { label: value, value }
  }
}

function isSubset(set, iterable) {
  const iterableSet = new Set(iterable)
  for (const elem of set) {
    if (!iterableSet.has(elem)) {
      return false
    }
  }
  return true
}

const HOMOZYGOTE_GT_OPTIONS = new Set([
  '1',
  //
  '1/1',
  //
  '1|1',
])

const HETEROZYGOTE_GT_OPTIONS = new Set([
  '0/1',
  '1/0',
  //
  '0|1',
  '1|0',
  //
  '1|2',
])

const GT_OPTIONS = [...HETEROZYGOTE_GT_OPTIONS, ...HOMOZYGOTE_GT_OPTIONS].map(toTagOption)

// TODO: Refactor
function GenotypeFilter(props) {
  const hasAllHeterozygoteOptions = isSubset(HETEROZYGOTE_GT_OPTIONS, props.gtFilterEvent)
  const hasAllHomozygoteOptions = isSubset(HOMOZYGOTE_GT_OPTIONS, props.gtFilterEvent)

  const getTagOptions = () => {
    let tags = []
    if (hasAllHeterozygoteOptions && hasAllHomozygoteOptions) {
      tags = ['Homozygote', 'Heterozygote']
    } else if (hasAllHeterozygoteOptions) {
      const rest = props.gtFilterEvent.filter((o) => !HETEROZYGOTE_GT_OPTIONS.has(o))
      tags = ['Heterozygote', ...rest]
    } else if (hasAllHomozygoteOptions) {
      const rest = props.gtFilterEvent.filter((o) => !HOMOZYGOTE_GT_OPTIONS.has(o))
      tags = ['Homozygote', ...rest]
    } else {
      tags = props.gtFilterEvent
    }

    return tags.map(toTagOption)
  }

  const getCategory = () => {
    if (hasAllHeterozygoteOptions && hasAllHomozygoteOptions) {
      return 'ANY'
    } else if (hasAllHeterozygoteOptions) {
      return 'NO'
    } else if (hasAllHomozygoteOptions) {
      return 'YES'
    } else {
      return 'ANY'
    }
  }

  const handleCategoryChange = (newCategory) => {
    switch (newCategory) {
      case 'ANY':
        props.setGtFilterEvent([...HOMOZYGOTE_GT_OPTIONS, ...HETEROZYGOTE_GT_OPTIONS])
        break
      // Homozygote
      case 'YES':
        props.setGtFilterEvent([...HOMOZYGOTE_GT_OPTIONS])
        break
      // Heterozygote
      case 'NO':
        props.setGtFilterEvent([...HETEROZYGOTE_GT_OPTIONS])
        break
      default:
      // do nothing
    }
  }

  const handleGtTagChange = (newData) => {
    const newValues = flatMap((option) => {
      switch (option.value) {
        case 'Heterozygote':
          return [...HETEROZYGOTE_GT_OPTIONS]
        case 'Homozygote':
          return [...HOMOZYGOTE_GT_OPTIONS]
        default:
          return option.value
      }
    }, newData)
    props.setGtFilterEvent(newValues)
  }

  return (
    <Stack direction="column" spacing={2}>
      <YesNoAnyButtonGroup
        value={getCategory()}
        title="Category"
        onChange={(value) => handleCategoryChange(value)}
        copy={{
          any: 'All',
          yes: 'Homozygote',
          no: 'Heterozygote',
        }}
      />
      <Tags
        title="Genotype"
        options={GT_OPTIONS.filter(({ value }) => !props.gtFilterEvent.includes(value))}
        value={getTagOptions()}
        onChange={handleGtTagChange}
      />
    </Stack>
  )
}

export default GenotypeFilter
