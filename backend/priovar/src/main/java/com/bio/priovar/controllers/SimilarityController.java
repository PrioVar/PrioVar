// similarity report controller
// Compare this snippet from backend/priovar/src/main/java/com/bio/priovar/controllers/SimilarityReportController.java:
package com.bio.priovar.controllers;

import com.bio.priovar.models.Patient;
import com.bio.priovar.models.SimilarityReport;
import com.bio.priovar.services.PatientService;
import com.bio.priovar.services.SimilarityService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/similarityReport")
@CrossOrigin
public class SimilarityController {

    // similarity service
    private final SimilarityService similarityService;

    // patient service
    private final PatientService patientService;

    // constructor
    @Autowired
    public SimilarityController(SimilarityService similarityService, PatientService patientService) {
        this.similarityService = similarityService;
        this.patientService = patientService;
    }

    // get all similarity reports
    @GetMapping()
    public List<SimilarityReport> getAllSimilarityReports() {
        return similarityService.getAllSimilarityReports();
    }

    // get similarity report by id
    @GetMapping("/{similarityReportId}")
    public SimilarityReport getSimilarityReportById(@PathVariable("similarityReportId") Long id) {
        return similarityService.findSimilarityReportById(id);
    }

    // get all similarity reports by patient id
    @GetMapping("/byPatient/{patientId}")
    public List<SimilarityReport> getAllSimilarityReportsByPatientId(@PathVariable("patientId") Long id) {
        return similarityService.findAllSimilarityReportsByPatientId(id);
    }

    @GetMapping("/byPatient/{patientId}/{numberOfReports}")
    public List<SimilarityReport> getSimilarityReportsByPatientId(@PathVariable("patientId") Long id, @PathVariable("numberOfReports") int numberOfReports) {
        return similarityService.findNewestSimilarityReportsByPatientId(id, numberOfReports);
    }

    // find the most similar patient
    @GetMapping("/mostSimilarPatient/{patientId}")
    public Patient getMostSimilarPatient(@PathVariable("patientId") Long id) {
        return similarityService.findMostSimilarPatientByCosine(id);
    }

    // find the most similar patientS (multiple) using findMostSimilarPatientsByCosine by specifying the number of patients
    @GetMapping("/mostSimilarPatients/{patientId}/{numberOfPatients}")
    public SimilarityReport getMostSimilarPatients(@PathVariable("patientId") Long id, @PathVariable("numberOfPatients") int numberOfPatients) {
        return similarityService.findMostSimilarPatientsByCosine(id, numberOfPatients);
    }

}