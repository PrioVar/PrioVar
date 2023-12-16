package com.bio.priovar.controllers;

import com.bio.priovar.repositories.ClinicianRepository;
import com.bio.priovar.repositories.MedicalCenterRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/initialize")
public class InitializerController {
    private final MedicalCenterRepository medicalCenterRepository;
    private final ClinicianRepository clinicianRepository;

    @Autowired
    public InitializerController(MedicalCenterRepository medicalCenterRepository, ClinicianRepository clinicianRepository) {
        this.medicalCenterRepository = medicalCenterRepository;
        this.clinicianRepository = clinicianRepository;
    }

    @PostMapping()
    public ResponseEntity<String> initialize() {

        return ResponseEntity.ok("Initialized Succesfully!");
    }
}
