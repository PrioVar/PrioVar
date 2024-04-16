import axios from '../../utils/axios'
import { API_BASE_URL } from '../../constants'

export const deleteVcfFile = async (fileId) => {
  const { data } = await axios.delete(`${API_BASE_URL}/vcf/${fileId}`)

  return data
}
