import { useQuery } from 'react-query'
import axios from '../../utils/axios'
import { API_BASE_URL } from '../../constants'

const fetchTrio = async (fileId) => {
  const { data } = await axios.get(`${API_BASE_URL}/vcf/${fileId}/trio`)

  return data
}

export const updateTrio = async (fileId, { mother, father }) => {
  const dataToUpload = {
    //    file: fileId,
    mother_file: mother.fileId || null,
    //  mother_sample_name: mother.sample || null,
    father_file: father.fileId || null,
    // father_sample_name: father.sample || null,
  }

  const { data } = await axios.patch(`${API_BASE_URL}/vcf/${fileId}/trio`, dataToUpload)

  return data
}

export const useTrio = ({ fileId }) => {
  return useQuery(['trio', fileId], () => fetchTrio(fileId))
}
