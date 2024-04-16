import axios from '../../utils/axios'
import { API_BASE_URL } from '../../constants'
import useOptimisticQuery from '../../hooks/useOptimisticQuery'

const fetchGenePhenotypeList = async (geneId) => {
  const { data } = await axios.get(`${API_BASE_URL}/gene/${geneId}`)
  return data?.phenotype_list
}

const updateGenePhenotypeList = async (geneId, phenotypeList) => {
  const { data } = await axios.patch(`${API_BASE_URL}/gene/${geneId}`, {
    phenotype_list: phenotypeList,
  })
  return data
}

export const useGenePhenotypeList = ({ geneId }) => {
  const [rawValue = [], setRawValue] = useOptimisticQuery({
    queryKey: ['gene-phenotype', geneId],
    queryFn: () => fetchGenePhenotypeList(geneId),
    mutationFn: (phenotypeList) => updateGenePhenotypeList(geneId, phenotypeList),
    delay: 0,
  })

  const value = rawValue.map((phenotype) => ({ label: phenotype, value: phenotype }))
  const setValue = (newOptions) => {
    const newValues = newOptions.map((phenotype) => phenotype.value)
    setRawValue(newValues)
  }

  return [value, setValue]
}

export const hpoIdtoGeneName = async (hpoId) => {
  const { data: firstData } = await axios.get(`https://hpo.jax.org/api/hpo/term/${hpoId}/genes?max=${1}`)
  const limit = firstData.geneCount
  const { data } = await axios.get(`https://hpo.jax.org/api/hpo/term/${hpoId}/genes?max=${limit}`)
  const geneNames = [...data.genes.map((gene) => gene.geneSymbol)]
  return geneNames
}
