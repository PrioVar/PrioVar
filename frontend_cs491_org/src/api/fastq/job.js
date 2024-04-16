import axios from '../../utils/axios'
import { API_BASE_URL } from '../../constants'

export const startJobsForFastqFile = async (pairId, options) => {
  const { cnv, alignment, snp } = options
  const { data } = await axios.post(`${API_BASE_URL}/fastq/${pairId}/job`, {
    options: { cnv, alignment, snp },
  })

  return data
}
