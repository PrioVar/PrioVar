package com.bio.priovar.services;

import com.bio.priovar.models.MedicalCenter;
import com.bio.priovar.models.Patient;
import com.bio.priovar.repositories.MedicalCenterRepository;
import com.bio.priovar.repositories.PatientRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class PatientService {

    private final PatientRepository patientRepository;
    private final MedicalCenterRepository medicalCenterRepository;

    @Autowired
    public PatientService(PatientRepository patientRepository, MedicalCenterRepository medicalCenterRepository) {
        this.patientRepository = patientRepository;
        this.medicalCenterRepository = medicalCenterRepository;
    }
    public List<Patient> getAllPatients() {
        return patientRepository.findAll();
    }

    public Patient getPatientById(Long patientId) {
        return patientRepository.findById(patientId).get();
    }

    public void addPatient(Patient patient) {
        patientRepository.save(patient);
    }

    public List<Patient> getPatientsByMedicalCenterId(Long medicalCenterId) {
        Optional<MedicalCenter> medicalCenterOptional = medicalCenterRepository.findById(medicalCenterId);

        if ( !medicalCenterOptional.isPresent() ) {
            throw new IllegalStateException("Medical center with id: " + medicalCenterId + " doesn't exist!");
        }

        return patientRepository.findAllByMedicalCenter_ID(medicalCenterId);
    }
}
