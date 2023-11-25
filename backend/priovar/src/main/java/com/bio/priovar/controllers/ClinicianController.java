package com.bio.priovar.controllers;

import com.bio.priovar.models.Clinician;
import com.bio.priovar.services.ClinicianService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/clinician")
@CrossOrigin
public class ClinicianController {
    private final ClinicianService clinicianService;

    @Autowired
    public ClinicianController(ClinicianService clinicianService) {
        this.clinicianService = clinicianService;
    }

    @GetMapping()
    public List<Clinician> getAllClinicians() {
        return clinicianService.getAllClinicians();
    }

    @GetMapping("/{clinicianId}")
    public Clinician getClinicianById(@PathVariable("clinicianId") Long id) {
        return clinicianService.getClinicianById(id);
    }

    @PostMapping("/add")
    public ResponseEntity<String> addClinician(@RequestBody Clinician clinician) {
        return new ResponseEntity<>(clinicianService.addClinician(clinician), org.springframework.http.HttpStatus.OK);
    }
}
