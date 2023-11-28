import { useQuery } from 'react-query'
import axios from '../../utils/axios'
import { API_BASE_URL } from '../../constants'

const fetchUserList = async () => {
  const { data } = await axios.get(`${API_BASE_URL}/person`)

  return data.map(({ user, organisation, ...rest }) => ({
    ...user,
    ...rest,
  }))
}
export const useUserList = () => {
  return useQuery(['user-list'], () => fetchUserList())
}
