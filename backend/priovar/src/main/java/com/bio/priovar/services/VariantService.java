package com.bio.priovar.services;

import com.bio.priovar.models.Variant;
import com.bio.priovar.repositories.VariantRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

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
}
