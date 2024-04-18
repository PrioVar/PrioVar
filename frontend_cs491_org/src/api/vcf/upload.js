import { isCancel } from 'axios'
import axios from '../../utils/axios'
import { API_BASE_URL } from '../../constants'
import { PATH_AUTH, PATH_PrioVar, ROOTS_PrioVar } from '../../routes/paths'

export const uploadFile = async (file, onUploadProgress, cancelToken) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);

    reader.onload = async () => {
      const base64EncodedFile = reader.result;

      try {
        console.log("Im tryn.s.d.sdsa")
        const response = await axios.post(`${ROOTS_PrioVar}/vcf/`, JSON.stringify(base64EncodedFile), {
          headers: {
            'Content-Type': 'application/json'
          },
          onUploadProgress: progressEvent => {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            onUploadProgress(percentCompleted);
          },
          cancelToken: cancelToken
        });

        resolve(response.data);
      } catch (error) {
        if (!isCancel(error)) {
          reject(error);
        }
      }
    };

    reader.onerror = error => reject(error);
  });
};
