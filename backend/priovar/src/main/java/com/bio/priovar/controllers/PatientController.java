package com.bio.priovar.controllers;

import com.bio.priovar.models.Patient;
import com.bio.priovar.models.PhenotypeTerm;
import com.bio.priovar.models.dto.PatientDTO;
import com.bio.priovar.models.dto.PatientWithPhenotype;
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
    public List<PatientDTO> getPatientsByMedicalCenterId(@PathVariable("medicalCenterId") Long medicalCenterId) {
        return patientService.getPatientsByMedicalCenterId(medicalCenterId);
    }

    @GetMapping("/byClinician/{clinicianId}")
    public List<PatientDTO> getPatientsByClinicianId(@PathVariable("clinicianId") Long clinicianId) {
        return patientService.getPatientsByClinicianId(clinicianId);
    }
    /*@GetMapping("/allAvailables/{medicalCenterId}")
    public List<PatientDTO> getAllAvailablePatients(@PathVariable("medicalCenterId") Long medicalCenterId) {
        return patientService.getAllAvailablePatients(medicalCenterId);
    }*/

    @GetMapping("/requested/{medicalCenterId}")
    public List<PatientDTO> getRequestedPatients(@PathVariable("medicalCenterId") Long medicalCenterId) {

        return patientService.getRequestedPatients(medicalCenterId);
    }

    @GetMapping("/allAvailable/{medicalCenterId}")
    public List<PatientDTO> getAllAvailablePatients(@PathVariable("medicalCenterId") Long medicalCenterId) {
        return patientService.getAllAvailablePatients(medicalCenterId);
    }

    @GetMapping("/byGenesusId/{genesusId}")
    public Patient getPatientByGenesusId(@PathVariable("genesusId") String genesusId) {
        return patientService.getPatientByGenesusId(genesusId);
    }

    @GetMapping("/termsByFileName/{fileName}")
    public List<String> getPatientPhenotypeTermsByFileName(@PathVariable("fileName") String fileName) {
        return patientService.getPatientByFileName(fileName);
    }

    @GetMapping("/getPatient")
    public Patient getPatientForDetailedView() {
        return patientService.getPatientForDetailedView();
    }

    @GetMapping("/phenotypeTerms/{patientId}")
    public List<PhenotypeTerm> getPhenotypeTermsOfPatient(@PathVariable("patientId") Long patientId) {
        return patientService.getPhenotypeTermsOfPatient(patientId);
    }

    @PostMapping("/add")
    public ResponseEntity<String> addPatient(@RequestBody Patient patient) {
        return new ResponseEntity<>(patientService.addPatient(patient), patient.getMedicalCenter() == null ? org.springframework.http.HttpStatus.BAD_REQUEST : org.springframework.http.HttpStatus.OK);
    }

    @PostMapping("/addPatientWithPhenotype")
    public ResponseEntity<String> addPatientWithPhenotype(@RequestBody PatientWithPhenotype patientDto) {
        return patientService.addPatientWithPhenotype(patientDto);
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

    @PostMapping("/{patientId}/addGene/{geneId}")
    public ResponseEntity<String> addGeneToPatient(@PathVariable("patientId") Long patientId, @PathVariable("geneId") Long geneId) {
        return new ResponseEntity<>(patientService.addGeneToPatient(patientId, geneId), org.springframework.http.HttpStatus.OK);
    }

    @PostMapping("/setVCF")
    public ResponseEntity<String> setVCFFileOfPatient(@RequestParam("patientId") Long patientId, 
                                                    @RequestParam("vcfFileId") Long vcfFileId) {   
        System.out.println("patientId: " + patientId + " vcfFileId: " + vcfFileId);
        System.out.println("patientId: " + patientId + " vcfFileId: " + vcfFileId);
        System.out.println("patientId: " + patientId + " vcfFileId: " + vcfFileId);

        return patientService.setVCFFileOfPatient(patientId, vcfFileId);
    }

    @PostMapping("/phenotypeTerm/{patientId}")
    public ResponseEntity<String> addPhenotypeTermFromPatientByPhenotypeTermId( @PathVariable("patientId") Long patientId, @RequestParam List<Long> phenotypeTermIds ) {
        return new ResponseEntity<>( patientService.addPhenotypeTermFromPatientByPhenotypeTermId(patientId, phenotypeTermIds), org.springframework.http.HttpStatus.OK);
    }
    
    

    // Delete Requests

    @DeleteMapping("/{patientId}")
    public ResponseEntity<String> deletePatient(@PathVariable("patientId") Long patientId) {
        return new ResponseEntity<>(patientService.deletePatient(patientId), org.springframework.http.HttpStatus.OK);
    }
    // delete phenotypeTerm from the patient
    @DeleteMapping("/{patientId}/deletePhenotypeTerm/{phenotypeTermId}")
    public ResponseEntity<String> deletePhenotypeTermFromPatient(@PathVariable("patientId") Long patientId, @PathVariable("phenotypeTermId") Long phenotypeTermId) {
        return new ResponseEntity<>(patientService.deletePhenotypeTermFromPatient(patientId, phenotypeTermId), org.springframework.http.HttpStatus.OK);
    }

    @DeleteMapping("/phenotypeTerm/{patientId}/{phenotypeTermId}")
    public ResponseEntity<String> deletePhenotypeTermFromPatientByPhenotypeTermId(@PathVariable("patientId") Long patientId, @PathVariable("phenotypeTermId") Long phenotypeTermId) {
        return new ResponseEntity<>(patientService.deletePhenotypeTermFromPatientByPhenotypeTermId(patientId, phenotypeTermId), org.springframework.http.HttpStatus.OK);
    }
}
