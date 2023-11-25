package com.bio.priovar.services;

import com.bio.priovar.models.Clinician;
import com.bio.priovar.models.MedicalCenter;
import com.bio.priovar.repositories.ClinicianRepository;
import com.bio.priovar.repositories.MedicalCenterRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class ClinicianService {
    private final ClinicianRepository clinicianRepository;
    private final MedicalCenterRepository medicalCenterRepository;

    @Autowired
    public ClinicianService(ClinicianRepository clinicianRepository, MedicalCenterRepository medicalCenterRepository) {
        this.clinicianRepository = clinicianRepository;
        this.medicalCenterRepository = medicalCenterRepository;
    }

    public List<Clinician> getAllClinicians() {
        return clinicianRepository.findAll();
    }

    public String addClinician(Clinician clinician) {
        MedicalCenter medicalCenter = clinician.getMedicalCenter();

        if ( medicalCenter == null ) {
            return "Medical Center is required";
        }

        Long medicalCenterId = medicalCenter.getId();
        medicalCenter = medicalCenterRepository.findById(medicalCenterId).orElse(null);

        if ( medicalCenter == null ) {
            return "Medical Center with id " + medicalCenterId + " does not exist";
        }

        clinician.setMedicalCenter(medicalCenter);
        clinicianRepository.save(clinician);
        return "Clinician added successfully";
    }

    public Clinician getClinicianById(Long id) {
        return clinicianRepository.findById(id).orElse(null);
    }
}
