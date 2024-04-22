package com.bio.priovar.services;

import com.bio.priovar.models.*;
import com.bio.priovar.repositories.VCFRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;


@Service
public class VCFService {
    private final VCFRepository vcfRepository;
    private final ClinicianService clinicianService;


    @Autowired
    public VCFService(VCFRepository vcfRepository, ClinicianService clinicianService) {
        this.clinicianService = clinicianService;
        this.vcfRepository = vcfRepository;
    }

    public ResponseEntity<String> uploadVCF(String base64Data, Long clinicianId) {

        VCFFile vcfFile = new VCFFile();
        vcfFile.setContent(base64Data);

        // generate an uuid file name
        String fileName = UUID.randomUUID().toString();
        vcfFile.setFileName(fileName);

        vcfFile.setClinician(clinicianService.getClinicianById(clinicianId));

        List<ClinicianComment> clinicianComments = new ArrayList<>();
        vcfFile.setClinicianComments(clinicianComments);

        vcfRepository.save(vcfFile);

        return ResponseEntity.ok("VCF File uploaded successfully");
    }
}
