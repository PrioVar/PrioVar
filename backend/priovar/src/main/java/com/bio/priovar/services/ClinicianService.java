package com.bio.priovar.services;

import com.bio.priovar.models.Clinician;
import com.bio.priovar.models.MedicalCenter;
import com.bio.priovar.models.Patient;
import com.bio.priovar.models.dto.LoginObject;
import com.bio.priovar.repositories.ClinicianRepository;
import com.bio.priovar.repositories.MedicalCenterRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
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

    public void addPatientToClinician(Long clinicianId, Patient patient) {
        Clinician clinician = clinicianRepository.findById(clinicianId).orElse(null);
        if ( clinician == null ) {
            System.out.println("Clinician with id " + clinicianId + " does not exist");
            return;
        }
        //Check if the patient list of clinician is null, else create it then add the patient and save the clinician
        if ( clinician.getPatients() == null ) {
            List<Patient> patients = new ArrayList<>();
            clinician.setPatients(patients);
        }
        clinician.getPatients().add(patient);
        clinicianRepository.save(clinician);
    }

    public MedicalCenter getClinicianMedicalCenterByClinicianId(Long clinicianId) {
        Clinician clinician = clinicianRepository.findById(clinicianId).orElse(null);
        if ( clinician == null ) {
            System.out.println("Clinician with id " + clinicianId + " does not exist");
            return null;
        }
        return clinician.getMedicalCenter();
    }


    public ResponseEntity<LoginObject> loginClinician(String email, String password) {
        Clinician clinician = clinicianRepository.findByEmail(email);
        LoginObject loginObject = new LoginObject();

        if ( clinician == null ) {
            loginObject.setMessage("Clinician with email " + email + " does not exist");
            loginObject.setId(-1L);
            return ResponseEntity.badRequest().body(loginObject);
        }

        if ( !clinician.getPassword().equals(password) ) {
            loginObject.setMessage("Incorrect password");
            loginObject.setId(-1L);
            return ResponseEntity.badRequest().body(loginObject);
        }

        loginObject.setMessage("Login successful");
        loginObject.setId(clinician.getId());
        loginObject.setRelatedId(clinician.getMedicalCenter().getId());
        return ResponseEntity.ok(loginObject);
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

    public List<Patient> getAllPatientsByClinicianId(Long clinicianId) {
        Clinician clinician = clinicianRepository.findById(clinicianId).orElse(null);

        if ( clinician == null ) {
            System.out.println("Clinician with id " + clinicianId + " does not exist");
            return null;
        }
        System.out.println("Clinician ID " + clinicianId + " requested all patients in service");
        return clinician.getPatients();
    }

    public List<Clinician> getAllCliniciansByMedicalCenterId(Long medicalCenterId) {
        return clinicianRepository.findAllByMedicalCenterId(medicalCenterId);
    }

    public String getClinicianNameById(Long id) {
        Clinician clinician = clinicianRepository.findById(id).orElse(null);
        if (clinician == null) {
            return null;
        }
        return clinician.getName();
    }

    public ResponseEntity<String> addRequestedPatientToClinician(Clinician clinician, Patient patient) {
        if ( clinician.getRequestedPatients() == null ) {
            clinician.setRequestedPatients(new ArrayList<>());
        }
        //Add the patient to the requested patients list of the clinician if it does not already exist
        if ( clinician.getRequestedPatients().contains(patient) ) {
            return ResponseEntity.badRequest().body("Patient already requested");
        }
        clinician.getRequestedPatients().add(patient);
        clinicianRepository.save(clinician);
        return ResponseEntity.ok("Requested patient added successfully");

    }
}
