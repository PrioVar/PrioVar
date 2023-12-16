package com.bio.priovar.services;

import com.bio.priovar.models.Clinician;
import com.bio.priovar.models.MedicalCenter;
import com.bio.priovar.repositories.ClinicianRepository;
import com.bio.priovar.repositories.MedicalCenterRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
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

    public ResponseEntity<String> addClinician(Clinician clinician) {
        MedicalCenter medicalCenter = clinician.getMedicalCenter();

        if ( medicalCenter == null ) {
            return ResponseEntity.badRequest().body("Medical Center is required");
        }

        Long medicalCenterId = medicalCenter.getId();
        medicalCenter = medicalCenterRepository.findById(medicalCenterId).orElse(null);

        if ( medicalCenter == null ) {
            return ResponseEntity.badRequest().body("Medical Center with id " + medicalCenterId + " does not exist");
        }

        Clinician clinicianWithSameEmail = clinicianRepository.findByEmail(clinician.getEmail());
        if ( clinicianWithSameEmail != null ) {
            return ResponseEntity.badRequest().body("Clinician with email " + clinician.getEmail() + " already exists");
        }

        clinician.setMedicalCenter(medicalCenter);
        clinicianRepository.save(clinician);
        return ResponseEntity.ok("Clinician added successfully");
    }

    public Clinician getClinicianById(Long id) {
        return clinicianRepository.findById(id).orElse(null);
    }

    public ResponseEntity<String> loginClinician(String email, String password) {
        Clinician clinician = clinicianRepository.findByEmail(email);

        if ( clinician == null ) {
            return ResponseEntity.badRequest().body("Clinician with email " + email + " does not exist");
        }

        if ( !clinician.getPassword().equals(password) ) {
            return ResponseEntity.badRequest().body("Incorrect password");
        }

        return ResponseEntity.ok("Login successful");
    }

    public ResponseEntity<String> changePasswordByEmailClinician(String email, String newPass, String oldPass) {
        Clinician clinician = clinicianRepository.findByEmail(email);

        if ( clinician == null ) {
            return ResponseEntity.badRequest().body("Clinician with email " + email + " does not exist");
        }

        if ( !clinician.getPassword().equals(oldPass) ) {
            return ResponseEntity.badRequest().body("Incorrect password");
        }

        if ( newPass.equals(oldPass) ) {
            return ResponseEntity.badRequest().body("New password cannot be same as old password");
        }

        clinician.setPassword(newPass);
        clinicianRepository.save(clinician);
        return ResponseEntity.ok("Password changed successfully");
    }
}
