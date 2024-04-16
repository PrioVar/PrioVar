import React from 'react'
import { HPO_EXTENDED_OPTIONS } from '../../../constants'
import Tags from '../../../components/Tags'
import { useGenePhenotypeList } from '../../../api/gene'

const CustomPhenotypes = function ({ geneId }) {
  const [phenotypeList, setPhenotypeList] = useGenePhenotypeList({ geneId })

  return (
    <Tags
      title={`Custom Phenotypes of ${geneId}`}
      options={HPO_EXTENDED_OPTIONS}
      value={phenotypeList}
      onChange={setPhenotypeList}
      sortKeys={['id', 'label']}
      freeSolo
      sx={{ width: 'fit-content', minWidth: 300 }}
    />
  )
}

export default CustomPhenotypes
