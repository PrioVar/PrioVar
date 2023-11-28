import { useMutation, useQuery, useQueryClient } from 'react-query'
import axios from '../../utils/axios'
import { API_BASE_URL } from '../../constants'

const fetchSampleMetadata = async (fileId) => {
  const { data } = await axios.get(`${API_BASE_URL}/vcf/${fileId}/redux`)

  return data?.redux_state
}
const updateSampleMetadata = async (fileId, sampleName, tables) => {
  await axios.patch(`${API_BASE_URL}/vcf/${fileId}/redux`, { redux_state: { tables } })
}
export const useSampleMetadata = ({ fileId, sampleName }) => {
  const queryKey = ['sample-metadata', fileId]
  const queryClient = useQueryClient()
  const query = useQuery(queryKey, () => fetchSampleMetadata(fileId))

  const mutation = useMutation(async (updater) => {
    await queryClient.cancelQueries(queryKey)

    const newTables = updater(query.data?.tables ?? [])
    // Optimistically update the current state first
    queryClient.setQueryData(queryKey, () => ({ tables: newTables }))
    // Then update the data on the server side
    updateSampleMetadata(fileId, sampleName, newTables)
  })

  const addNewTable = (newTable) => mutation.mutate((currentTables) => [...currentTables, newTable])
  const removeTable = (tableId) =>
    mutation.mutate((currentTables) => currentTables.filter((table) => table.id !== tableId))
  const updateTable = (tableId, data) =>
    mutation.mutate((currentTables) => currentTables.map((table) => (table.id === tableId ? data : table)))

  return {
    query,
    addNewTable,
    removeTable,
    updateTable,
  }
}
