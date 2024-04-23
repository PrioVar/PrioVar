package com.bio.priovar.controllers;

import com.bio.priovar.models.Clinician;
import com.bio.priovar.models.Patient;
import com.bio.priovar.models.dto.LoginObject;
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
        System.out.println("Clinician ID " + id + " requested in controller");
        return clinicianService.getClinicianById(id);
    }
    
    @GetMapping("getName/{clinicianId}")
    public String getClinicianNameById(@PathVariable("clinicianId") Long id) {
        System.out.println("Clinician ID " + id + " requested in controller");
        return clinicianService.getClinicianNameById(id);
    }
    
    @GetMapping("/byMedicalCenter/{medicalCenterId}")
    public List<Clinician> getAllCliniciansByMedicalCenterId(@PathVariable("medicalCenterId") Long medicalCenterId) {
        return clinicianService.getAllCliniciansByMedicalCenterId(medicalCenterId);
    }

    @GetMapping("/allPatients/{clinicianId}")
    public List<Patient> getAllPatientsByClinicianId(@PathVariable("clinicianId") Long clinicianId) {
        System.out.println("Clinician ID " + clinicianId + " requested all patients in controller");
        return clinicianService.getAllPatientsByClinicianId(clinicianId);
    }

    @PostMapping("/add")
    public ResponseEntity<String> addClinician(@RequestBody Clinician clinician) {
        return clinicianService.addClinician(clinician);
    }

    @PostMapping("/login")
    public ResponseEntity<LoginObject> loginClinician(@RequestParam String email, @RequestParam String password ) {
        System.out.println(email + " " + password);
        return clinicianService.loginClinician(email,password);
    }

    @PatchMapping("/changePassword")
    public ResponseEntity<String> changePasswordClinician(@RequestParam String email, @RequestParam String newPass, @RequestParam String oldPass) {
        return clinicianService.changePasswordByEmailClinician(email,newPass, oldPass);
    }
}
