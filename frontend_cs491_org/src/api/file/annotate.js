import axios from '../../utils/axios'
import { API_BASE_URL } from '../../constants'

export const annotateFile = async (fileId, annotation, type) => {
  const { data } = await axios.post(`${API_BASE_URL}/${type === 'VCF' ? 'vcf' : 'fastq'}/annotate`, {
    fileId,
    annotation,
  })

  return data
}

export const getPlots = async (fileId) => {
  const { data } = await axios.get(`${API_BASE_URL}/fastq/plots/${fileId}`)

  return data
}
