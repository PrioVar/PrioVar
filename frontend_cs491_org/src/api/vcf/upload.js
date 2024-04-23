import { isCancel } from 'axios'
import axios from '../../utils/axios'
import { API_BASE_URL } from '../../constants'
import { PATH_AUTH, PATH_PrioVar, ROOTS_PrioVar } from '../../routes/paths'

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

        // Assuming response.data contains the ID of the uploaded VCF file as a Long
        if (response && response.data) {
            localStorage.setItem('vcfFileId', response.data.toString()); 
            console.log(response.data.toString());
            resolve(response.data);
        } else {
        reject(new Error('No data in response'));
        }
      } catch (error) {
        if (!isCancel(error)) {
          reject(error);
        }
      }
    };

    reader.onerror = error => reject(error);
  });
};
