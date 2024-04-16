import { isCancel } from 'axios'
import EMA from 'src/utils/ema'
import axios from '../../utils/axios'

export const makeUploadFile = (url) => async (file, cancelToken, onUploadProgress) => {
  try {
    let lastTime = Date.now()
    let lastLoaded = 0
    const emaDt = new EMA(0.1)
    const emaDl = new EMA(0.1)

    const { data } = await axios.post(url, file, {
      headers: {
        'Content-Type': file.type,
        'Content-Disposition': `attachment; filename="${file.name}"`,
      },
      onUploadProgress(e) {
        const dt = emaDt.update(new Date() - lastTime)
        const dl = emaDl.update(e.loaded - lastLoaded)
        const speed = dl / dt
        lastLoaded = e.loaded
        lastTime = new Date()

        const percentCompleted = Math.round((e.loaded * 100) / e.total)
        const timeRemaining = Math.round((e.total - e.loaded) / speed / 1000)
        onUploadProgress({ percentCompleted, timeRemaining })
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
