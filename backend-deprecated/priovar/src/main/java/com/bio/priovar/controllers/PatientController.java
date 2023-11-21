package com.bio.priovar.controllers;

import com.bio.priovar.models.Patient;
import com.bio.priovar.services.PatientService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping({"/patient"})
@CrossOrigin
public class PatientController {

    private final PatientService patientService;

    @Autowired
    public PatientController(PatientService patientService) {
        this.patientService = patientService;
    }

    @GetMapping()
    public List<Patient> getAllPatients() {
        return patientService.getAllPatients();
    }

    @GetMapping("{patientId}")
    public Patient getPatientById(@PathVariable("patientId") Long patientId) {
        return patientService.getPatientById(patientId);
    }

    @GetMapping("/medicalCenter/{medicalCenterId}")
    public List<Patient> getPatientsByMedicalCenterId(@PathVariable("medicalCenterId") Long medicalCenterId) {
        return patientService.getPatientsByMedicalCenterId(medicalCenterId);
    }

    @PostMapping("/add")
    public void addPatient(@RequestBody Patient patient) {
        patientService.addPatient(patient);
    }
}
