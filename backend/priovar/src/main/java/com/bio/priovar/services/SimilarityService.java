package com.bio.priovar.services;//service for similarity

import com.bio.priovar.models.Patient;
import com.bio.priovar.models.SimilarityReport;
import com.bio.priovar.models.SimilarityStrategy;
import com.bio.priovar.models.BasicHPOCosineSimilarity;

import com.bio.priovar.repositories.SimilarityReportRepository;
import com.bio.priovar.services.PatientService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
public class SimilarityService {
    private SimilarityReportRepository similarityReportRepository;

    //patient service
    private PatientService patientService;

    //constructor
    @Autowired
    public SimilarityService(SimilarityReportRepository similarityReportRepository, PatientService patientService) {
        this.similarityReportRepository = similarityReportRepository;
        this.patientService = patientService;
    }

    public SimilarityReport findSimilarityReportById(Long id) {
        return similarityReportRepository.findSimilarityReportById(id);
    }

    public List<SimilarityReport> findAllSimilarityReportsByPrimaryPatientId(Long id) {
        return similarityReportRepository.findAllSimilarityReportsByPrimaryPatientId(id);
    }

    public List<SimilarityReport> findAllSimilarityReportsByPatientId(Long id) {
        return similarityReportRepository.findAllSimilarityReportsByPatientId(id);
    }

    public void save(SimilarityReport similarityReport) {
        similarityReportRepository.save(similarityReport);
    }

    public void delete(SimilarityReport similarityReport) {
        similarityReportRepository.delete(similarityReport);
    }

    //find similar patients
    public List<Patient> getSimilarPatientsByBasicCosine(long primaryPatientId, float treshold){

        // get all patients from patient service using get all patients
        List<Patient> allPatients = patientService.getAllPatients();

        // for all patients calculate similarity
        SimilarityStrategy basicCosine = new BasicHPOCosineSimilarity();

        // get primary patient
        Patient primaryPatient = patientService.getPatientById(primaryPatientId);

        //patinet list to return
        List<Patient> similarPatients = new ArrayList<Patient>();

        // get all similarity scores above treshold
        for (Patient patient : allPatients) {

            if (patient.getId() == primaryPatientId) {
                continue;
            }

            float similarityScore = basicCosine.calculateSimilarity(primaryPatient, patient);
            if (similarityScore >= treshold) {
                System.out.println("Similarity score: " + similarityScore);
                System.out.println("Patient: " + patient.getName());

                //cretae similarity report
                SimilarityReport similarityReport = new SimilarityReport();
                similarityReport.setPrimaryPatient(primaryPatient);
                similarityReport.setSecondaryPatient(patient);
                similarityReport.setTotalScore(similarityScore);
                similarityReport.setSimilarityStrategy(basicCosine);
                similarityReport.setStatus("pending");

                //save similarity report
                save(similarityReport);

                //add patient to similar patients list
                similarPatients.add(patient);
            }
        }

        return similarPatients;

    }

    //find the most similar patient
    public Patient findMostSimilarPatientByCosine(Long primaryPatientId){

        //from all patients find the most similar patient
        List<Patient> allPatients = patientService.getAllPatients();

        //get primary patient
        Patient primaryPatient = patientService.getPatientById(primaryPatientId);

        // vectoize if needed
        if(primaryPatient.getPhenotypeVector() == null){
            patientService.vectorizePatientPhenotype(primaryPatientId);
        }
        primaryPatient = patientService.getPatientById(primaryPatientId);

        //similarity strategy
        SimilarityStrategy basicCosine = new BasicHPOCosineSimilarity();

        //most similar patient
        Patient mostSimilarPatient = null;

        //similarity score
        float similarityScore = -2;

        for (Patient patient : allPatients) {
            if (patient.getId().equals(primaryPatientId)) {
                continue;
            }


            // vectorize if needed
            if(patient.getPhenotypeVector() == null){
                String response = patientService.vectorizePatientPhenotype(patient.getId());
                System.out.println(response);
            }

            patient = patientService.getPatientById(patient.getId());


            float currentSimilarityScore = basicCosine.calculateSimilarity(primaryPatient, patient);
            if (currentSimilarityScore > similarityScore) {
                similarityScore = currentSimilarityScore;
                mostSimilarPatient = patient;
            }
        }


        return mostSimilarPatient;
    }


    public List<SimilarityReport> getAllSimilarityReports() {
        return similarityReportRepository.findAll();
    }
}
