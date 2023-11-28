import axios from '../../utils/axios'
import { API_BASE_URL } from '../../constants'

export const getCoverage = async (fastqId, geneName, coverage) => {
  const { data } = await axios.get(`${API_BASE_URL}/coverage/${fastqId}/${geneName}/${coverage}`)

  return data
}
