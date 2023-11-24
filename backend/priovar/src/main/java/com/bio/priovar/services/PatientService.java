package com.bio.priovar.services;

import com.bio.priovar.models.Disease;
import com.bio.priovar.models.Patient;
import com.bio.priovar.models.Variant;
import com.bio.priovar.repositories.DiseaseRepository;
import com.bio.priovar.repositories.MedicalCenterRepository;
import com.bio.priovar.repositories.PatientRepository;
import com.bio.priovar.repositories.VariantRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class PatientService {
    private final PatientRepository patientRepository;
    private final DiseaseRepository diseaseRepository;
    private final MedicalCenterRepository medicalCenterRepository;
    private final VariantRepository variantRepository;

    @Autowired
    public PatientService(PatientRepository patientRepository, DiseaseRepository diseaseRepository, MedicalCenterRepository medicalCenterRepository, VariantRepository variantRepository) {
        this.patientRepository = patientRepository;
        this.diseaseRepository = diseaseRepository;
        this.medicalCenterRepository = medicalCenterRepository;
        this.variantRepository = variantRepository;
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
}
