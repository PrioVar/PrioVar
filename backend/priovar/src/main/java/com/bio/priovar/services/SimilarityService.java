package com.bio.priovar.services;//service for similarity

import com.bio.priovar.models.Patient;
import com.bio.priovar.models.SimilarityReport;
import com.bio.priovar.models.SimilarityStrategy;
import com.bio.priovar.models.BasicHPOCosineSimilarity;

import com.bio.priovar.repositories.SimilarityReportRepository;
import com.bio.priovar.services.PatientService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.util.Pair;
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
                similarityReport.setSimilarityStrategy("BasicHPOCosineSimilarity");
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

    // find most similar patientS by cosine
    public List<SimilarityReport> findMostSimilarPatientsByCosine(Long primaryPatientId, int numberOfPatients) {

        //from all patients find the most similar patient
        List<Patient> allPatients = patientService.getAllPatients();

        //get primary patient
        Patient primaryPatient = patientService.getPatientById(primaryPatientId);

        // vectoize if needed
        if (primaryPatient.getPhenotypeVector() == null) {
            patientService.vectorizePatientPhenotype(primaryPatientId);
        }
        primaryPatient = patientService.getPatientById(primaryPatientId);

        //similarity strategy
        SimilarityStrategy basicCosine = new BasicHPOCosineSimilarity();

        //most similar patient list which is sorted
        List<Pair<Patient, Float>> mostSimilarPatientsAndScores = new ArrayList<Pair<Patient, Float>>();

        //similarity score
        float similarityScore = -2;

        for (Patient patient : allPatients) {
            if (patient.getId().equals(primaryPatientId)) {
                continue;
            }

            // vectorize if needed
            if (patient.getPhenotypeVector() == null) {
                String response = patientService.vectorizePatientPhenotype(patient.getId());
                System.out.println(response);
            }

            patient = patientService.getPatientById(patient.getId());

            float currentSimilarityScore = basicCosine.calculateSimilarity(primaryPatient, patient);

            //find the position of the patient in the sorted list
            int position = 0;
            for (Pair<Patient, Float> pair : mostSimilarPatientsAndScores) {
                if (pair.getSecond() < currentSimilarityScore) {
                    break;
                }
                position++;
            }

            //add the patient to the list if the list is not full
            if (mostSimilarPatientsAndScores.size() < numberOfPatients) {
                mostSimilarPatientsAndScores.add(position, Pair.of(patient, currentSimilarityScore));
            } else {
                //if the list is full, add the patient only if the similarity score is higher than the lowest score in the list
                if( position < numberOfPatients) {
                    mostSimilarPatientsAndScores.add(position, Pair.of(patient, currentSimilarityScore));
                    mostSimilarPatientsAndScores.remove(numberOfPatients);
                }
            }
        }

        // create similarity reports
        List<SimilarityReport> similarityReports = new ArrayList<SimilarityReport>();

        for (Pair<Patient, Float> pair : mostSimilarPatientsAndScores) {
            Patient patient = pair.getFirst();
            float currentSimilarityScore = pair.getSecond();

            //cretae similarity report
            SimilarityReport similarityReport = new SimilarityReport();
            similarityReport.setPrimaryPatient(primaryPatient);
            similarityReport.setSecondaryPatient(patient);
            similarityReport.setTotalScore(currentSimilarityScore);
            similarityReport.setSimilarityStrategy("BasicHPOCosineSimilarity");
            similarityReport.setStatus("pending");
            similarityReport.setPhenotypeScore(currentSimilarityScore);


            Boolean isReportExist = false;
            //check if the patient pair has already been calculated by getting the similarity reportss
            List<SimilarityReport> similarityReports1 = similarityReportRepository.findAllSimilarityReportsByPatientId(primaryPatientId);
            for (SimilarityReport similarityReport1 : similarityReports1) {
                if (similarityReport1.getSecondaryPatient().getId().equals(patient.getId()) || similarityReport1.getPrimaryPatient().getId().equals(patient.getId()) ){
                    //if their similarity strategy is the same, then just update the phenotype score and total score
                    if (similarityReport1.getSimilarityStrategy().equals("BasicHPOCosineSimilarity")) {
                        similarityReport1.setPhenotypeScore(currentSimilarityScore);
                        similarityReport1.setTotalScore(currentSimilarityScore);
                        similarityReportRepository.save(similarityReport1);
                        isReportExist = true;
                        break;
                    }
                }
            }

            //save similarity report
            if (!isReportExist) {
                save(similarityReport);
            }

            //add similarity report to the list
            similarityReports.add(similarityReport);
        }

        return similarityReports;
    }




    public List<SimilarityReport> getAllSimilarityReports() {
        return similarityReportRepository.findAll();
    }
}
