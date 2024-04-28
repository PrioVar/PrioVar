import { useSelector } from 'react-redux'
import { useQuery } from 'react-query'
import axios from '../../utils/axios'
import { API_BASE_URL } from '../../constants'
import { equals } from 'ramda'
import { useDebounce } from 'use-debounce'

const translateApiFiltersToBackend = (apiFilters) => {
  return Object.values(apiFilters)
    .filter((filter) => filter.enabled && !equals(filter.initialValue, filter.value))
    .map((filter) => ({
      name: filter.backendName,
      value: filter.value,
    }))
}
const fetchVariants = async (fileId, sampleName, page, pageSize, sortBy, sortDirection, filters) => {
  //console.log(1)
  //console.log(sortBy)
  //console.log(sortDirection)
  //console.log(filters)
  const { data } = await axios.post(`${API_BASE_URL}/variants/${fileId}/${sampleName}`, {
    page: page + 1,
    page_size: pageSize,
    sort_by: sortBy,
    sort_direction: sortDirection,
    filters,
  })

  return data
}
// FIXME: Re-renders every second
export const useVariants = ({ fileId, sampleName, page = 0, pageSize = 10, sortBy = 0, sortDirection = 'asc' }) => {
  const apiFilters = useSelector((state) => state)
  const translatedFilters = translateApiFiltersToBackend(apiFilters)
  const [debouncedApiFilters, ] = useDebounce(translatedFilters, 1000, { equalityFn: equals })

  return useQuery(['variants', fileId, sampleName, page, pageSize, sortBy, sortDirection, debouncedApiFilters], () =>
    fetchVariants(fileId, sampleName, page, pageSize, sortBy, sortDirection, debouncedApiFilters),
  )
}
