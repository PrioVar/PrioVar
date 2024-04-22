import { useQuery, useQueryClient } from 'react-query'
import axios from '../../utils/axios'
import { API_BASE_URL,  } from '../../constants'
import { ROOTS_PrioVar } from '../../routes/paths'

const fetchAllFiles = async () => {
  const { data } = await axios.get(`${API_BASE_URL}/file/list`)

  return data
}

const fetchBedFiles = async () => {
  const { data } = await axios.get(`${API_BASE_URL}/bed/list`)
  return data
}

// NEW ADDITION ERKIN
const fecthMedicalCenterFiles = async () => {
  const { data } = await axios.get(`${ROOTS_PrioVar}/file/list/medical-center`)
  return data
}


export const useFiles = () => {
  const queryClient = useQueryClient()

  const query = useQuery(['all-files'], fetchAllFiles, {
    // TODO: Fix
    // refetchInterval: 1000 * 15, // 15 seconds
    // refetchIntervalInBackground: 1000 * 60, // 1 minute
    refetchOnWindowFocus: true,
  })
  const refresh = async () => {
    console.log('REFRESHING')
    await queryClient.refetchQueries('all-files')
    await queryClient.resetQueries('all-files')
  }

  return { query, refresh }
}

export const useFileDetails = (fileId) => {
  const queryClient = useQueryClient()
  const query = useQuery(['file-details', fileId], async () => {
    const { data } = await axios.get(`${API_BASE_URL}/file/${fileId}`)
    return data
  })
  const refresh = async () => {
    console.log('REFRESHING')
    await queryClient.refetchQueries('file-details')
    await queryClient.resetQueries('file-details')
  }

  return { query, refresh }
}

export const useBedFiles = () => {
  const queryClient = useQueryClient()
  const query = useQuery(['bed-files'], fetchBedFiles, {
    refetchOnWindowFocus: true,
  })
  const refresh = async () => {
    console.log('REFRESHING')
    await queryClient.refetchQueries('bed-files')
    await queryClient.resetQueries('bed-files')
  }

  return { query, refresh }
}

export const updateFinishInfo = async (fileId) => {
  const { data } = await axios.put(`${API_BASE_URL}/file/${fileId}`, {
    type: 'finish',
  })
  return data
}

export const updateFileNotes = async (fileId, notes) => {
  const { data } = await axios.put(`${API_BASE_URL}/file/${fileId}`, {
    type: 'notes',
    notes,
  })
  return data
}
