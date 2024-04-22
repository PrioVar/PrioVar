package com.bio.priovar.controllers;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import com.bio.priovar.models.VCFFile;
import com.bio.priovar.models.dto.VCFFileDTO;
import com.bio.priovar.services.VCFService;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;


@RestController
@RequestMapping("/vcf")
@CrossOrigin
public class VCFController {
    private final VCFService vcfService;

    @Autowired
    public VCFController(VCFService vcfService) {
        this.vcfService = vcfService;
    }

    @PostMapping("/upload")
    public ResponseEntity<Long> uploadVCF(@RequestParam("vcfFile") String base64File, 
                                        @RequestParam("clinicianId") Long clinicianId,
                                        @RequestParam("medicalCenterId") Long medicalCenterId){
        // Remove the Base64 prefix if present (e.g., "data:image/png;base64,")
        String base64Data = base64File.substring(base64File.indexOf(",") + 1);
        return vcfService.uploadVCF(base64Data, clinicianId, medicalCenterId);
    }

    @GetMapping("/byMedicalCenter/{medicalCenterId}")
    public List<VCFFileDTO> getVCFFilesByMedicalCenterId(@PathVariable("medicalCenterId") Long medicalCenterId) {
        return vcfService.getVCFFilesByMedicalCenterId(medicalCenterId);
    }

    @GetMapping("/byClinician/{clinicianId}")
    public List<VCFFileDTO> getVCFFilesByClinicianId(@PathVariable("clinicianId") Long clinicianId) {
        return vcfService.getVCFFilesByClinicianId(clinicianId);
    }
}