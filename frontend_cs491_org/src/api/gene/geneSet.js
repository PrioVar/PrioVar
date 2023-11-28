import axios from '../../utils/axios'
import { API_BASE_URL } from '../../constants'

export const fetchGeneSet = async () => {
  const { data } = await axios.get(`${API_BASE_URL}/genesets`)
  const result = data?.map((geneset) => {
    return { value: geneset.name, label: `${geneset.name} - ${geneset.organisation}`, custom: true }
  })
  return result
}

export const updateGeneSet = async (name, content) => {
  const { data } = await axios.post(`${API_BASE_URL}/genesets`, {
    name: name,
    genes: content,
  })
  return data
}

export const removeGeneSet = async (name) => {
  const { data } = await axios.delete(`${API_BASE_URL}/genesets/${name}`)
  return data
}
