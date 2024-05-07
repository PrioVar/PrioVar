package com.bio.priovar.services;

import com.bio.priovar.models.Variant;
import com.bio.priovar.repositories.VariantRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class VariantService {
    private final VariantRepository variantRepository;

    @Autowired
    public VariantService(VariantRepository variantRepository) {
        this.variantRepository = variantRepository;
    }

    public String addVariant(Variant variant) {
        System.out.println(variant.getAlt());
        variantRepository.save(variant);
        return "Variant added successfully";
    }

    public List<Variant> getAllVariants() {
        return variantRepository.findAll();
    }

    public Variant getVariantById(Long id) {
        return variantRepository.findById(id).orElse(null);
    }

    public List<Variant> getVariantsByPatientId(Long patientId) {
        return variantRepository.getVariantsByPatientId(patientId);
    }
}
