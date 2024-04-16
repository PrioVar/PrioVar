import { makeUploadFile } from '../utils'
import { API_BASE_URL } from '../../constants'

export const uploadFile = makeUploadFile(`${API_BASE_URL}/fastq/`)
