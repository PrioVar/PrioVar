import axios from '../../utils/axios'
import { API_BASE_URL } from '../../constants'
import useOptimisticQuery from '../../hooks/useOptimisticQuery'

const fetchFileTitle = async (fileId) => {
  const { data } = await axios.get(`${API_BASE_URL}/vcf/${fileId}/title`)
  return data?.title ?? ''
}
const updateFileTitle = async (fileId, title) => {
  const { data } = await axios.patch(`${API_BASE_URL}/vcf/${fileId}/title`, { title })
  return data
}
export const useFileTitle = ({ fileId }) => {
  return useOptimisticQuery({
    queryKey: ['file-title', fileId],
    queryFn: () => fetchFileTitle(fileId),
    mutationFn: (title) => updateFileTitle(fileId, title),
    delay: 250,
  })
}
