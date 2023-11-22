package com.bio.priovar.controllers;

import com.bio.priovar.models.MedicalCenter;
import com.bio.priovar.services.MedicalCenterService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/medicalCenter")
@CrossOrigin
public class MedicalCenterController {
    private final MedicalCenterService medicalCenterService;

    @Autowired
    public MedicalCenterController(MedicalCenterService medicalCenterService) {
        this.medicalCenterService = medicalCenterService;
    }

    @PostMapping("/add")
    public void addMedicalCenter(@RequestBody MedicalCenter medicalCenter) {
        medicalCenterService.addMedicalCenter(medicalCenter);
    }
}
