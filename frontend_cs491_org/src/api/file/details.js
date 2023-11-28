import axios from '../../utils/axios'
import { API_BASE_URL } from '../../constants'

export const updateDetails = async (fileId, details, type) => {
  const { data } = await axios.post(`${API_BASE_URL}/file/details`, {
    fileId,
    details,
    type,
  })

  return data
}
