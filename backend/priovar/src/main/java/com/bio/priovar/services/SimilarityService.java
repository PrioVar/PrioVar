package com.bio.priovar.services;//service for similarity

import com.bio.priovar.models.*;

import com.bio.priovar.repositories.PairSimilarityRepository;
import com.bio.priovar.repositories.SimilarityReportRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.util.Pair;
import org.springframework.stereotype.Service;

import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.List;

@Service
public class SimilarityService {
    private PairSimilarityRepository pairSimilarityRepository;
    private SimilarityReportRepository similarityReportRepository;

    private PatientService patientService;

    //constructor
    @Autowired
    public SimilarityService(PairSimilarityRepository pairSimilarityRepository, SimilarityReportRepository similarityReportRepository, PatientService patientService) {
        this.pairSimilarityRepository = pairSimilarityRepository;
        this.similarityReportRepository = similarityReportRepository;
        this.patientService = patientService;
    }

    public SimilarityReport findSimilarityReportById(Long id) {
        return similarityReportRepository.findSimilarityReportById(id);
    }

    public List<SimilarityReport> findAllSimilarityReportsByPatientId(Long id) {
        return similarityReportRepository.findAllByPrimaryPatientId(id);
    }

    public void save(SimilarityReport similarityReport) {
        similarityReportRepository.save(similarityReport);
    }

    public void delete(SimilarityReport similarityReport) {
        similarityReportRepository.delete(similarityReport);
    }

    // find similar patients
    public List<Patient> getSimilarPatientsByBasicCosine(long primaryPatientId, float treshold){

        // get all patients from patient service using get all patients
        List<Patient> allPatients = patientService.getAllPatients();

        // for all patients calculate similarity
        SimilarityStrategy basicCosine = new BasicHPOCosineSimilarity();

        // get primary patient
        Patient primaryPatient = patientService.getPatientById(primaryPatientId);

        // patient list to return
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


                PairSimilarity pairSimilarity = new PairSimilarity();
                pairSimilarity.setPrimaryPatient(primaryPatient);
                pairSimilarity.setSecondaryPatient(patient);
                pairSimilarity.setTotalScore(similarityScore);
                pairSimilarity.setStatus(PairSimilarity.REPORT_STATUS.PENDING);
                pairSimilarity.setPhenotypeScore(similarityScore);
                pairSimilarity.setSimilarityStrategy("BasicHPOCosineSimilarity");

                // save pair similarity
                pairSimilarityRepository.save(pairSimilarity);

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
    public SimilarityReport findMostSimilarPatientsByCosine(Long primaryPatientId, int numberOfPatients) {

        //from all patients find the most similar patient
        List<Patient> allPatients = patientService.getAllPatients();

        //get primary patient
        Patient primaryPatient = patientService.getPatientById(primaryPatientId);

        // vectorize if needed
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

        SimilarityReport similarityReport = new SimilarityReport();
        similarityReport.setPrimaryPatient(primaryPatient);
        similarityReport.setCreatedAt(OffsetDateTime.now(ZoneOffset.ofHours(3)));
        List<PairSimilarity> pairSimilarities = new ArrayList<PairSimilarity>();

        for (Pair<Patient, Float> pair : mostSimilarPatientsAndScores) {
            Patient patient = pair.getFirst();
            float currentSimilarityScore = pair.getSecond();

            PairSimilarity pairSimilarity = new PairSimilarity();
            pairSimilarity.setPrimaryPatient(primaryPatient);
            pairSimilarity.setSecondaryPatient(patient);
            pairSimilarity.setTotalScore(currentSimilarityScore);
            pairSimilarity.setStatus(PairSimilarity.REPORT_STATUS.PENDING);
            pairSimilarity.setPhenotypeScore(currentSimilarityScore);
            pairSimilarity.setSimilarityStrategy("BasicHPOCosineSimilarity");


            Boolean isReportExist = false;
            //check if the patient pair has already been calculated by getting the similarity reports
            List<PairSimilarity> pairSimilarities1 = pairSimilarityRepository.findAllPairSimilaritiesByPatientId(primaryPatientId);
            for ( PairSimilarity pairSimilarity1 : pairSimilarities1 ) {
                if ( pairSimilarity1.getSecondaryPatient().getId().equals(patient.getId()) || pairSimilarity1.getPrimaryPatient().getId().equals(patient.getId()) ){
                    //if their similarity strategy is the same, then just update the phenotype score and total score
                    if (pairSimilarity1.getSimilarityStrategy().equals("BasicHPOCosineSimilarity")) {
                        pairSimilarity1.setPhenotypeScore(currentSimilarityScore);
                        pairSimilarity1.setTotalScore(currentSimilarityScore);
                        pairSimilarityRepository.save(pairSimilarity1);
                        isReportExist = true;
                        break;
                    }
                }
            }

            //save similarity report
            if (!isReportExist) {
                pairSimilarityRepository.save(pairSimilarity);
            }

            //add similarity report to the list
            pairSimilarities.add(pairSimilarity);
        }

        similarityReport.setPairSimilarities(pairSimilarities);
        save(similarityReport);

        return similarityReport;
    }




    public List<SimilarityReport> getAllSimilarityReports() {
        return similarityReportRepository.findAll();
    }

    public List<SimilarityReport> findNewestSimilarityReportsByPatientId(Long id, int numberOfReports) {
        // find the newest numberOfReports similarity reports by patient id using createdAt
        List<SimilarityReport> similarityReports = similarityReportRepository.findAllByPrimaryPatientId(id);

        // sort the PairSimilarities inside each similarity report by totalScore in descending order
        for (SimilarityReport similarityReport : similarityReports) {
            similarityReport.getPairSimilarities().sort((o1, o2) -> Float.compare(o2.getTotalScore(), o1.getTotalScore()));
        }

        // sort the similarity reports by createdAt
        similarityReports.sort((o1, o2) -> o2.getCreatedAt().compareTo(o1.getCreatedAt()));

        // return the first numberOfReports similarity reports
        return similarityReports.subList(0, Math.min(numberOfReports, similarityReports.size()));
    }
}
