import { isCancel } from 'axios'
import axios from '../../utils/axios'
import { API_BASE_URL } from '../../constants'

export const uploadFile = async (file, onUploadProgress, cancelToken, name, ref) => {
  try {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('name', name)
    fd.append('ref', ref)
    const { data } = await axios.post(`${API_BASE_URL}/bed`, fd, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress(progressEvent) {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
        onUploadProgress(percentCompleted)
      },
      cancelToken,
    })

    return data
  } catch (error) {
    if (!isCancel(error)) {
      throw error
    }
  }
}
