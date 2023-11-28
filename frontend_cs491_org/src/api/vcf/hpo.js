// TODO: use mutation
import useOptimisticQuery from '../../hooks/useOptimisticQuery'
import axios from '../../utils/axios'
import { API_BASE_URL, HPO_MAP } from '../../constants'

const updateHpo = async (fileId, hpoIdList) => {
  const { data } = await axios.patch(`${API_BASE_URL}/vcf/${fileId}/hpo`, { hpo_id_list: hpoIdList })

  return data
}

const fetchHpo = async (fileId) => {
  const { data } = await axios.get(`${API_BASE_URL}/vcf/${fileId}/hpo`)

  return data?.hpo_id_list
}

export const useHpo = ({ fileId }) => {
  const [hpoIdList = [], setHpoIdList] = useOptimisticQuery({
    queryKey: ['hpo-id-list', fileId],
    queryFn: () => fetchHpo(fileId),
    mutationFn: (hpoIdList) => updateHpo(fileId, hpoIdList),
    delay: 0,
  })

  const hpoList = hpoIdList.map((hpoId) => HPO_MAP[hpoId])
  const setHpoList = (newHpoList) => {
    const newIds = newHpoList.map((hpo) => hpo.value)
    setHpoIdList(newIds)
  }

  return [hpoList, setHpoList]
}
