import { useQuery, useQueryClient } from 'react-query'
import axios from '../../utils/axios'
import { API_BASE_URL,  } from '../../constants'
import { ROOTS_PrioVar } from '../../routes/paths'

// Carried Here from MyPatients.jsx
export const fetchCurrentClinicianName = async () => {
  const data = await axios.get(`${ROOTS_PrioVar}/clinician/getName/${localStorage.getItem('clinicianId')}`)
  return data
}

export const deletePatient = async (patientId) => {
  const data = await axios.delete(`${ROOTS_PrioVar}/patient/${patientId}`)
  return data
}

export const deleteVCF = async (vcfId) => {
  const data = await axios.delete(`${ROOTS_PrioVar}/vcf/${vcfId}`)
  return data
}

export const addPatientWithPhenotype = async (request) => {
  const data = await axios.post(`${ROOTS_PrioVar}/patient/addPatientWithPhenotype`, request);
  return data
}

export const deletePhenotypeTerm = async ( patientId, phenotypeTermId) => {
  const data = await axios.delete(`${ROOTS_PrioVar}/patient/phenotypeTerm/${patientId}/${phenotypeTermId}`)
  return data
}

export const addPhenotypeTerm = async (patientId, phenotypeTermIds) => {
  const data = await axios.post(`${ROOTS_PrioVar}/patient/phenotypeTerm/${patientId}`, phenotypeTermIds)
  return data
}

// NEW ADDITION ERKIN
export const fetchClinicianPatients = async () => {
  const { data } = await axios.get(`${ROOTS_PrioVar}/patient/byClinician/${localStorage.getItem('clinicianId')}`)
  return data
}

// NEW ADDITION ERKIN
export const fecthMedicalCenterPatients = async () => {
  const { data } = await axios.get(`${ROOTS_PrioVar}/patient/byMedicalCenter/${localStorage.getItem('healthCenterId')}`)
  return data
}

export const fetchPatientVariants = async (patientId) => {
  const data = await axios.get(`${ROOTS_PrioVar}/variant/patient/${patientId}`)
  return data
}

export const fetchRequestedMedicalCenterPatients = async () => {
  const { data } = await axios.get(`${ROOTS_PrioVar}/patient/requested/${localStorage.getItem('healthCenterId')}`)
  return data
}

export const fetchAllAvailablePatients = async () => {
  const { data } = await axios.get(`${ROOTS_PrioVar}/patient/allAvailable/${localStorage.getItem('healthCenterId')}`)
  return data
}

// NEW ADDITION ERKIN
export const fecthClinicianFiles = async () => {
  const { data } = await axios.get(`${ROOTS_PrioVar}/vcf/byClinician/${localStorage.getItem('clinicianId')}`)
  return data
}

// NEW ADDITION ERKIN
export const fecthMedicalCenterFiles = async () => {
  const { data } = await axios.get(`${ROOTS_PrioVar}/vcf/byMedicalCenter/${localStorage.getItem('healthCenterId')}`)
  return data
}

export const fetchDiseases = async () => {
  const { data } = await axios.get(`${ROOTS_PrioVar}/disease`)
  return data
}

export const fetchPhenotypeTerms = async (patientId) => {
  const { data } = await axios.get(`${ROOTS_PrioVar}/patient/phenotypeTerms/${patientId}`)
  return data
}

const fetchAllFiles = async () => {
  const { data } = await axios.get(`${API_BASE_URL}/file/list`)

  return data
}

const fetchBedFiles = async () => {
  const { data } = await axios.get(`${API_BASE_URL}/bed/list`)
  return data
}

export const markNotificationAsRead = async (notificationId) => {
  const { data } = await axios.post(`${ROOTS_PrioVar}/notification/markRead/${notificationId}`);
  return data;
};

export const markAllNotificationsAsRead = async (actorId) => {
  const { data } = await axios.post(`${ROOTS_PrioVar}/notification/markAllRead/${actorId}`);
  return data;
};

// list.js

export const acceptInformationRequest = async (informationRequestId, responseMessage) => {
  try {
    console.log(responseMessage)
    const response = await axios.post(`${ROOTS_PrioVar}/request/accept/${informationRequestId}?notificationAppendix=${responseMessage}`);
    return response.data;
  } catch (error) {
    console.error('Failed to accept information request:', error);
    throw error;
  }
};

export const rejectInformationRequest = async (informationRequestId, responseMessage) => {
  try {
    const response = await axios.post(`${ROOTS_PrioVar}/request/reject/${informationRequestId}?notificationAppendix=${responseMessage}`);
    return response.data;
  } catch (error) {
    console.error('Failed to reject information request:', error);
    throw error;
  }
};

export const fetchWaitingInformationRequests = async (clinicianId) => {
  try {
    const response = await axios.get(`${ROOTS_PrioVar}/request/waiting/${clinicianId}`);
    return response.data;
  } catch (error) {
    console.error('Failed to get waiting information requests:', error);
    throw error;
  }
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

export const fetchNotifications = async (actorId) => {
  try {
    const response = await axios.get(`${ROOTS_PrioVar}/notification/listNotifications/${actorId}`);
    return response.data;
  } catch (error) {
    throw error;
  }
};

// list.js
export const sendInformationRequest = async (clinicianId, patientId, requestDescription) => {
  const params = new URLSearchParams({ clinicianId, patientId, requestDescription });
  try {
    const response = await axios.post(`${ROOTS_PrioVar}/request/send`, params);
    return response.data;
  } catch (error) {
    console.error("Failed to send information request:", error.response);
    throw error;
  }
};


export const updateFileNotes = async (vcfFileId, notes) => {
  const formData = new FormData();
  formData.append('vcfFileId', vcfFileId);
  formData.append('clinicianId', localStorage.getItem('clinicianId'))
  formData.append('clinicianNotes', notes)
  try {
    const { data } = await axios.post(`${ROOTS_PrioVar}/vcf/addNote`, formData);
    return data

  } catch (error) {
    console.log(error)
  }
  
  


  
  /*
  const { data } = await axios.put(`${API_BASE_URL}/file/${fileId}`, {
    type: 'notes',
    notes,
  })
  return data
  */
}
