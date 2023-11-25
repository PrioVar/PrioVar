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

    @GetMapping("/byMedicalCenter/{medicalCenterId}")
    public List<Patient> getPatientsByMedicalCenterId(@PathVariable("medicalCenterId") Long medicalCenterId) {
        return patientService.getPatientsByMedicalCenterId(medicalCenterId);
    }

    @GetMapping("/byClinician/{clinicianId}")
    public List<Patient> getPatientsByClinicianId(@PathVariable("clinicianId") Long clinicianId) {
        return patientService.getPatientsByClinicianId(clinicianId);
    }

    @PostMapping("/add")
    public ResponseEntity<String> addPatient(@RequestBody Patient patient) {
        return new ResponseEntity<>(patientService.addPatient(patient), patient.getMedicalCenter() == null ? org.springframework.http.HttpStatus.BAD_REQUEST : org.springframework.http.HttpStatus.OK);
    }

    @PostMapping("/addPatientToClinician/{clinicianId}")
    public ResponseEntity<String> addPatientToClinician(@RequestBody Patient patient, @PathVariable("clinicianId") Long clinicianId) {
        return new ResponseEntity<>(patientService.addPatientToClinician(patient, clinicianId), patient.getMedicalCenter() == null ? org.springframework.http.HttpStatus.BAD_REQUEST : org.springframework.http.HttpStatus.OK);
    }

    // add disease to the patient
    @PostMapping("/{patientId}/addDisease/{diseaseId}")
    public ResponseEntity<String> addDiseaseToPatient(@PathVariable("patientId") Long patientId, @PathVariable("diseaseId") Long diseaseId) {
        return new ResponseEntity<>(patientService.addDiseaseToPatient(patientId, diseaseId), org.springframework.http.HttpStatus.OK);
    }

    // add phenotypeTerm to the patient
    @PostMapping("/{patientId}/addPhenotypeTerm/{phenotypeTermId}")
    public ResponseEntity<String> addPhenotypeTermToPatient(@PathVariable("patientId") Long patientId, @PathVariable("phenotypeTermId") Long phenotypeTermId) {
        return new ResponseEntity<>(patientService.addPhenotypeTermToPatient(patientId, phenotypeTermId), org.springframework.http.HttpStatus.OK);
    }

    @PostMapping("/{patientId}/addVariant/{variantId}")
    public ResponseEntity<String> addVariantToPatient(@PathVariable("patientId") Long patientId, @PathVariable("variantId") Long variantId) {
        return new ResponseEntity<>(patientService.addVariantToPatient(patientId, variantId), org.springframework.http.HttpStatus.OK);
    }

    // Delete Requests

    // delete phenotypeTerm from the patient
    @DeleteMapping("/{patientId}/deletePhenotypeTerm/{phenotypeTermId}")
    public ResponseEntity<String> deletePhenotypeTermFromPatient(@PathVariable("patientId") Long patientId, @PathVariable("phenotypeTermId") Long phenotypeTermId) {
        return new ResponseEntity<>(patientService.deletePhenotypeTermFromPatient(patientId, phenotypeTermId), org.springframework.http.HttpStatus.OK);
    }
}
