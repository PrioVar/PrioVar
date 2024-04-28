import { isCancel } from 'axios'
import axios from '../../utils/axios'
import { ROOTS_PrioVar } from '../../routes/paths'

export const uploadFile = async (file, onUploadProgress, cancelToken) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);

    reader.onload = async () => {
      const vcfFile = reader.result;
      const formData = new FormData();
      formData.append('vcfFile', vcfFile);
      formData.append('clinicianId', localStorage.getItem('clinicianId'))
      formData.append('medicalCenterId', localStorage.getItem('healthCenterId'))
      try {
        const response = await axios.post(`${ROOTS_PrioVar}/vcf/upload`, formData, {
          
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
