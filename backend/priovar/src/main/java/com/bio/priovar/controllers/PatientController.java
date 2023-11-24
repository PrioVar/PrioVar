package com.bio.priovar.controllers;

import com.bio.priovar.models.Patient;
import com.bio.priovar.services.PatientService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/patient")
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

    @GetMapping("/{patientId}")
    public Patient getPatientById(@PathVariable("patientId") Long id) {
        return patientService.getPatientById(id);
    }

    @GetMapping("/byDisease/{diseaseName}")
    public List<Patient> getPatientsByDiseaseName(@PathVariable("diseaseName") String diseaseName) {
        return patientService.getPatientsByDiseaseName(diseaseName);
    }

    @PostMapping("/add")
    public ResponseEntity<String> addPatient(@RequestBody Patient patient) {
        return new ResponseEntity<>(patientService.addPatient(patient), patient.getMedicalCenter() == null ? org.springframework.http.HttpStatus.BAD_REQUEST : org.springframework.http.HttpStatus.OK);
    }

    // add disease to the patient
    @PostMapping("/{patientId}/addDisease/{diseaseId}")
    public ResponseEntity<String> addDiseaseToPatient(@PathVariable("patientId") Long patientId, @PathVariable("diseaseId") Long diseaseId) {
        return new ResponseEntity<>(patientService.addDiseaseToPatient(patientId, diseaseId), org.springframework.http.HttpStatus.OK);
    }
}
