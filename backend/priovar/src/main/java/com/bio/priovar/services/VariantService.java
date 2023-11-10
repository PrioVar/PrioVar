package com.bio.priovar.services;

import com.bio.priovar.models.Patient;
import com.bio.priovar.models.Variant;
import com.bio.priovar.repositories.PatientRepository;
import com.bio.priovar.repositories.VariantRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class VariantService {

    private final VariantRepository variantRepository;
    private final PatientRepository patientRepository;

    @Autowired
    public VariantService(VariantRepository variantRepository, PatientRepository patientRepository) {
        this.variantRepository = variantRepository;
        this.patientRepository = patientRepository;
    }

    public List<Variant> getAllVariants() {
        return variantRepository.findAll();
    }

    public Variant getVariantById(Long variantId) {
        return variantRepository.findById(variantId).get();
    }

    public void addVariant(Variant variant) {
        variantRepository.save(variant);
    }

    public List<Variant> getVariantsByPatientId(Long patientId) {
        // check if the patient exists
        Optional<Patient> patientOptional = patientRepository.findById(patientId);

        if ( !patientOptional.isPresent() ) {
            throw new IllegalStateException("Patient with id: " + patientId + " doesn't exist!");
        }

        return variantRepository.findAllByPatient_ID(patientId);
    }
}
