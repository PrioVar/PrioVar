package com.bio.priovar.controllers;

import com.bio.priovar.models.Variant;
import com.bio.priovar.services.VariantService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/variant")
@CrossOrigin
public class VariantController {
    private final VariantService variantService;

    @Autowired
    public VariantController(VariantService variantService) {
        this.variantService = variantService;
    }

    @PostMapping("/add")
    public ResponseEntity<String> addVariant(@RequestBody Variant variant) {
        return new ResponseEntity<>(variantService.addVariant(variant), org.springframework.http.HttpStatus.OK);
    }
}
