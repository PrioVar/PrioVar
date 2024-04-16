import { useQuery, useQueryClient } from 'react-query'
import axios from '../../utils/axios'
import { API_BASE_URL } from '../../constants'

const fetchAllFiles = async () => {
  const { data } = await axios.get(`${API_BASE_URL}/vcf/list`)

  return data
}
export const useFiles = () => {
  const queryClient = useQueryClient()

  const query = useQuery(['vcf-files'], fetchAllFiles, {
    // TODO: Fix
    // refetchInterval: 1000 * 15, // 15 seconds
    // refetchIntervalInBackground: 1000 * 60, // 1 minute
    refetchOnWindowFocus: true,
  })
  const refresh = () => queryClient.invalidateQueries('vcf-files')

  return { query, refresh }
}
