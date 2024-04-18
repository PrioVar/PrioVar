package com.bio.priovar.services;

import com.bio.priovar.repositories.VCFRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;


@Service
public class VCFService {
    private final VCFRepository vcfRepository;

    @Autowired
    public VCFService(VCFRepository vcfRepository) {
        this.vcfRepository = vcfRepository;
    }
}
