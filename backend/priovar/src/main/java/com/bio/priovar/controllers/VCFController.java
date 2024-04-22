package com.bio.priovar.controllers;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import com.bio.priovar.services.VCFService;
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
    public ResponseEntity<Long> uploadVCF(@RequestParam("vcfFile") String base64File, @RequestParam("clinicianId") Long clinicianId) {
        // Remove the Base64 prefix if present (e.g., "data:image/png;base64,")
        String base64Data = base64File.substring(base64File.indexOf(",") + 1);
        return vcfService.uploadVCF(base64Data, clinicianId);
    }

}