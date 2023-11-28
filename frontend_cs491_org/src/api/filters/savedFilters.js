import axios from '../../utils/axios'
import { API_BASE_URL } from '../../constants'

export const fetchSavedFilters = async () => {
  const { data } = await axios.get(`${API_BASE_URL}/saved-filters`)
  const result = data?.map((savedFilters) => {
    return {
      value: savedFilters.value,
      label: `${savedFilters.name} - ${savedFilters.organisation}`,
      name: savedFilters.name,
    }
  })
  return result
}

export const updateSavedFilter = async (name, content) => {
  const { data } = await axios.post(`${API_BASE_URL}/saved-filters`, {
    name: name,
    value: content,
  })
  return data
}

export const removeSavedFilter = async (name) => {
  const { data } = await axios.delete(`${API_BASE_URL}/saved-filters/${name}`)
  return data
}
