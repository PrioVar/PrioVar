package com.bio.priovar.controllers;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
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
    public ResponseEntity<String> uploadVCF(@RequestParam("vcfFile") String base64File, @RequestParam("clinicianId") Long clinicianId,
                                            @RequestParam("patientAge") int patientAge, @RequestParam("patientGender") String patientGender) {
        // Remove the Base64 prefix if present (e.g., "data:image/png;base64,")
        String base64Data = base64File.substring(base64File.indexOf(",") + 1);
        return vcfService.uploadVCF(base64Data, clinicianId, patientAge, patientGender);

    }

}