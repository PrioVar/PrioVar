package com.bio.priovar.controllers;

import org.springframework.web.bind.annotation.*;
import org.springframework.beans.factory.annotation.Autowired;
import com.bio.priovar.services.VCFService;

@RestController
@RequestMapping("/vcf")
@CrossOrigin
public class VCFController {
    private final VCFService vcfService;

    @Autowired
    public VCFController(VCFService vcfService) {
        this.vcfService = vcfService;
    }

    @PostMapping()
    public String uploadVCF(@RequestBody String base64EncodedFile) {
        // Decode the Base64 file and process
        System.out.println("File received in Base64 format");
        // You would typically decode this and process as a file here
        return "File uploaded successfully";
    }
}