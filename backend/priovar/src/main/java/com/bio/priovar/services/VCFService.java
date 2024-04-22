package com.bio.priovar.services;

import com.bio.priovar.models.*;
import com.bio.priovar.repositories.VCFRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;


@Service
public class VCFService {
    private final VCFRepository vcfRepository;
    private final ClinicianService clinicianService;
    private final MedicalCenterService medicalCenterService;

    @Autowired
    public VCFService(VCFRepository vcfRepository, ClinicianService clinicianService, MedicalCenterService medicalCenterService) {
        this.clinicianService = clinicianService;
        this.vcfRepository = vcfRepository;
        this.medicalCenterService = medicalCenterService;
    }

    public ResponseEntity<Long> uploadVCF(String base64Data, Long clinicianId, Long medicalCenterId) {

        VCFFile vcfFile = new VCFFile();
        vcfFile.setContent(base64Data);

        // generate an uuid file name
        String fileName = UUID.randomUUID().toString();
        vcfFile.setFileName(fileName);

        vcfFile.setClinician(clinicianService.getClinicianById(clinicianId));
        vcfFile.setMedicalCenter(medicalCenterService.getMedicalCenterById(medicalCenterId));

        List<ClinicianComment> clinicianComments = new ArrayList<>();
        vcfFile.setClinicianComments(clinicianComments);

        // Save the vcf file and get the saved entity with ID populated
        VCFFile savedVcfFile = vcfRepository.save(vcfFile);

        // Check if the saved entity is not null and has an ID
        if (savedVcfFile != null && savedVcfFile.getId() != null) {
            // Return the ID of the newly created file in the response
            return ResponseEntity.ok(savedVcfFile.getId());
        } else {
            // Handle the error case where the file wasn't saved correctly
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    public List<VCFFile> getVCFFilesByMedicalCenterId(Long medicalCenterId) {

        return vcfRepository.findAllByMedicalCenterId(medicalCenterId);
    }

}
