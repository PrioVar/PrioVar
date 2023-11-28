import axios from '../../utils/axios'
import { API_BASE_URL } from '../../constants'

export const startJobsForVcfFile = async (fileId) => {
  const { data } = await axios.post(`${API_BASE_URL}/vcf/${fileId}/job`)

  return data
}
