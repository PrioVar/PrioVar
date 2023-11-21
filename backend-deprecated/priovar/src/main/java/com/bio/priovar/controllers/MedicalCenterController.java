package com.bio.priovar.controllers;

import com.bio.priovar.models.MedicalCenter;
import com.bio.priovar.models.Patient;
import com.bio.priovar.services.MedicalCenterService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping({"/medicalCenter"})
@CrossOrigin
public class MedicalCenterController {

    private final MedicalCenterService medicalCenterService;

    @Autowired
    public MedicalCenterController(MedicalCenterService medicalCenterService) {
        this.medicalCenterService = medicalCenterService;
    }

    @GetMapping()
    public List<MedicalCenter> getAllMedicalCenters() {
        return medicalCenterService.getAllMedicalCenters();
    }

    @GetMapping("/{medicalCenterId}")
    public MedicalCenter getMedicalCenterById(@PathVariable("medicalCenterId") Long medicalCenterId) {
        return medicalCenterService.getMedicalCenterById(medicalCenterId);
    }

    @GetMapping("/patient/{medicalCenterId}")
    public List<Patient> getPatientsByMedicalCenterId(@PathVariable("medicalCenterId") Long medicalCenterId) {
        return medicalCenterService.getPatientsByMedicalCenterId(medicalCenterId);
    }

    @PostMapping("/add")
    public void addMedicalCenter(@RequestBody MedicalCenter medicalCenter) {
        medicalCenterService.addMedicalCenter(medicalCenter);
    }
}
