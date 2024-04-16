import { useQuery } from 'react-query'
import axios from '../../utils/axios'
import { API_BASE_URL } from '../../constants'

const fetchVcfFile = async (fileId) => {
  const { data } = await axios.get(`${API_BASE_URL}/vcf/${fileId}`)

  return data
}

export const useVcfFile = ({ fileId }) => {
  return useQuery(['vcf-file', fileId], () => fetchVcfFile(fileId))
}
