import { isCancel } from 'axios'
import axios from '../../utils/axios'
import { API_BASE_URL } from '../../constants'

export const uploadFile = async (file, onUploadProgress, cancelToken) => {
  try {
    const { data } = await axios.post(`${API_BASE_URL}/vcf/`, file, {
      headers: {
        'Content-Type': file.type,
        'Content-Disposition': `attachment; filename="${file.name}"`,
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
