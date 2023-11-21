package com.bio.priovar.controllers;

import com.bio.priovar.models.Variant;
import com.bio.priovar.services.VariantService;
import org.springframework.beans.factory.annotation.Autowired;
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

    @GetMapping("{variantId}")
    public Variant getVariantById(@PathVariable("variantId") Long variantId) {
        return variantService.getVariantById(variantId);
    }

    @GetMapping("/patientId/{patientId}")
    public List<Variant> getVariantsByPatientId(@PathVariable("patientId") Long patientId) {
        return variantService.getVariantsByPatientId(patientId);
    }

    @PostMapping("/add")
    public void addVariant(@RequestBody Variant variant) {
        variantService.addVariant(variant);
    }
}
