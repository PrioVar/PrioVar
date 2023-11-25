package com.bio.priovar.services;

import com.bio.priovar.models.*;
import com.bio.priovar.repositories.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class PatientService {
    private final PatientRepository patientRepository;
    private final DiseaseRepository diseaseRepository;
    private final MedicalCenterRepository medicalCenterRepository;
    private final VariantRepository variantRepository;
    private final ClinicianRepository clinicianRepository;

    @Autowired
    public PatientService(PatientRepository patientRepository, DiseaseRepository diseaseRepository, MedicalCenterRepository medicalCenterRepository, VariantRepository variantRepository, ClinicianRepository clinicianRepository) {
        this.patientRepository = patientRepository;
        this.diseaseRepository = diseaseRepository;
        this.medicalCenterRepository = medicalCenterRepository;
        this.variantRepository = variantRepository;
        this.clinicianRepository = clinicianRepository;
    }

    public String addPatient(Patient patient) {
        if ( patient.getDisease() != null ) {
            Long diseaseId = patient.getDisease().getId();
            patient.setDisease(diseaseRepository.findById(diseaseId).orElse(null));
        }

        if ( patient.getMedicalCenter() == null ) {
            // return an error
            return "Medical Center is required";
        }
        Long medicalCenterId = patient.getMedicalCenter().getId();
        patient.setMedicalCenter(medicalCenterRepository.findById(medicalCenterId).orElse(null));

        patientRepository.save(patient);
        return "Patient added successfully";
    }

    public List<Patient> getAllPatients() {
        return patientRepository.findAll();
    }

    public Patient getPatientById(Long id) {
        return patientRepository.findById(id).orElse(null);
    }

    public List<Patient> getPatientsByDiseaseName(String diseaseName) {
        return patientRepository.findByDiseaseName(diseaseName);
    }

    public String addDiseaseToPatient(Long patientId, Long diseaseId) {
        Disease disease = diseaseRepository.findById(diseaseId).orElse(null);
        Patient patient = patientRepository.findById(patientId).orElse(null);

        if ( disease == null ) {
            return "Disease not found";
        }

        if ( patient == null ) {
            return "Patient not found";
        }

        // if patient already has a disease, return an error
        if ( patient.getDisease() != null ) {
            return "Patient already has a disease";
        }

        patient.setDisease(disease);
        patientRepository.save(patient);
        return "Disease added to patient successfully";
    }

    public List<Patient> getPatientsByMedicalCenterId(Long medicalCenterId) {
        return patientRepository.findByMedicalCenterId(medicalCenterId);
    }

    public String addVariantToPatient(Long patientId, Long variantId) {
        // check if variant exists
        Variant variant = variantRepository.findById(variantId).orElse(null);
        Patient patient = patientRepository.findById(patientId).orElse(null);

        if ( variant == null ) {
            return "Variant not found";
        }

        // check if patient exists
        if ( patient == null ) {
            return "Patient not found";
        }

        // add the variant to the list of the patient if list is not empty, otherwise create a new list
        if ( patient.getVariants() != null ) {
            patient.getVariants().add(variant);
        } else {
            patient.setVariants(List.of(variant));
        }

        patientRepository.save(patient);
        return "Variant added to patient successfully";
    }

    public String addPatientToClinician(Patient patient, Long clinicianId) {
        if ( patient.getDisease() != null ) {
            Long diseaseId = patient.getDisease().getId();
            patient.setDisease(diseaseRepository.findById(diseaseId).orElse(null));
        }

        if ( patient.getMedicalCenter() == null ) {
            // return an error
            return "Medical Center is required";
        }
        Long medicalCenterId = patient.getMedicalCenter().getId();
        MedicalCenter medicalCenter = medicalCenterRepository.findById(medicalCenterId).orElse(null);

        if ( medicalCenter == null ) {
            return "Medical Center with id " + medicalCenterId + " does not exist";
        }

        Clinician clinician = clinicianRepository.findById(clinicianId).orElse(null);

        if ( clinician == null ) {
            return "Clinician with id " + clinicianId + " does not exist";
        }

        // add the patient to the list of the clinician if list is not empty, otherwise create a new list
        if ( clinician.getPatients() != null ) {
            clinician.getPatients().add(patient);
        } else {
            clinician.setPatients(List.of(patient));
        }

        clinicianRepository.save(clinician);
        patient.setMedicalCenter(medicalCenter);
        patientRepository.save(patient);
        return "Patient added successfully";
    }
}
