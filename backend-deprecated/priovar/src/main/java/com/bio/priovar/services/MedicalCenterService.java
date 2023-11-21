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
public class MedicalCenterService {

    private final MedicalCenterRepository medicalCenterRepository;
    private final PatientRepository patientRepository;

    @Autowired
    public MedicalCenterService(MedicalCenterRepository medicalCenterRepository, PatientRepository patientRepository) {
        this.medicalCenterRepository = medicalCenterRepository;
        this.patientRepository = patientRepository;
    }

    public List<MedicalCenter> getAllMedicalCenters() {
        return medicalCenterRepository.findAll();
    }

    public void addMedicalCenter(MedicalCenter medicalCenter) {
        // check if the name of the medical center already exists
        Optional<MedicalCenter> medicalCenterOptional = medicalCenterRepository.findByName(medicalCenter.getName());

        if ( medicalCenterOptional.isPresent() ) {
            throw new IllegalStateException("Medical center with name: " + medicalCenter.getName() + " already exists!");
        }

        medicalCenterRepository.save(medicalCenter);
    }

    public List<Patient> getPatientsByMedicalCenterId(Long medicalCenterId) {
        Optional<MedicalCenter> medicalCenterOptional = medicalCenterRepository.findById(medicalCenterId);

        if ( !medicalCenterOptional.isPresent() ) {
            throw new IllegalStateException("Medical center with id: " + medicalCenterId + " doesn't exist!");
        }

        return patientRepository.findAllByMedicalCenter_ID(medicalCenterId);
    }

    public MedicalCenter getMedicalCenterById(Long medicalCenterId) {
        Optional<MedicalCenter> medicalCenterOptional = medicalCenterRepository.findById(medicalCenterId);

        if ( !medicalCenterOptional.isPresent() ) {
            throw new IllegalStateException("Medical center with id: " + medicalCenterId + " doesn't exist!");
        }

        return medicalCenterOptional.get();
    }
}
