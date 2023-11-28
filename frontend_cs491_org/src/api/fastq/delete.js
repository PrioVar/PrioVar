import axios from '../../utils/axios'
import { API_BASE_URL } from '../../constants'

export const deleteFastqFile = async (fastqId) => {
  const { data } = await axios.delete(`${API_BASE_URL}/fastq/${fastqId}`)

  return data
}
