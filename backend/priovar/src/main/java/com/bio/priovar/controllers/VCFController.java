package com.bio.priovar.controllers;

import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

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

    @PostMapping("/upload")
    public String uploadVCF(@RequestParam("vcfFile") String base64File) {
        // Remove the Base64 prefix if present (e.g., "data:image/png;base64,")
        String base64Data = base64File.substring(base64File.indexOf(",") + 1);
        System.out.println(base64Data);
        return "File uploaded and saved successfully";

    }

}