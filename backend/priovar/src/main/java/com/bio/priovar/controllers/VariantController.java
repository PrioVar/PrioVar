package com.bio.priovar.controllers;

import com.bio.priovar.models.Variant;
import com.bio.priovar.services.VariantService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/variant")
@CrossOrigin
public class VariantController {
    private final VariantService variantService;

    @Autowired
    public VariantController(VariantService variantService) {
        this.variantService = variantService;
    }

    @GetMapping()
    public List<Variant> getAllVariants() {
        return variantService.getAllVariants();
    }

    @GetMapping("/{variantId}")
    public Variant getVariantById(@PathVariable("variantId") Long id) {
        return variantService.getVariantById(id);
    }

    @PostMapping("/add")
    public ResponseEntity<String> addVariant(@RequestBody Variant variant) {
        return new ResponseEntity<>(variantService.addVariant(variant), org.springframework.http.HttpStatus.OK);
    }

    // to get all variants of a patientid
    @GetMapping("/patient/{patientId}")
    public List<Variant> getVariantsByPatientId(@PathVariable("patientId") Long patientId) {
        return variantService.getVariantsByPatientId(patientId);
    }

}
