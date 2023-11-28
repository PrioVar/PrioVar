import { useMutation, useQuery, useQueryClient } from 'react-query'
import axios from '../../utils/axios'
import { API_BASE_URL } from '../../constants'

const deleteUser = async (userId) => {
  const { data } = await axios.delete(`${API_BASE_URL}/person/${userId}`)

  return data
}
const updateUser = async (userId, user) => {
  const { data } = await axios.patch(`${API_BASE_URL}/person/${userId}`, user)

  return data
}
const fetchUser = async (userId) => {
  const { data } = await axios.get(`${API_BASE_URL}/person/${userId}`)

  return data
}
const createUser = async (user) => {
  const { data } = await axios.post(`${API_BASE_URL}/person`, user)

  return data
}
const fetchSelf = async () => {
  const { data } = await axios.get(`${API_BASE_URL}/person/self`)

  return data
}

export const useManageUser = ({ id }) => {
  const queryClient = useQueryClient()

  const queryKey = ['user', id]

  const query = useQuery(queryKey, () => fetchUser(id))
  const self = useQuery('self', () => fetchSelf())

  const onSuccess = () => queryClient.invalidateQueries(['user-list'])
  const insertMutation = useMutation((user) => createUser(user), { onSuccess })
  const deleteMutation = useMutation(() => deleteUser(id), { onSuccess })
  const updateMutation = useMutation((user) => updateUser(id, user))

  return { self, query, insertMutation, deleteMutation, updateMutation }
}
