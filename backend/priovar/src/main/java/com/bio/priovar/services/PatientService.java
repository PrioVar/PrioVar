package com.bio.priovar.services;

import com.bio.priovar.models.*;
import com.bio.priovar.models.dto.PatientWithPhenotype;
import com.bio.priovar.repositories.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ClassPathResource;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class PatientService {
    private final PatientRepository patientRepository;
    private final DiseaseRepository diseaseRepository;
    private final MedicalCenterRepository medicalCenterRepository;
    private final VariantRepository variantRepository;
    private final ClinicianRepository clinicianRepository;
    private final PhenotypeTermRepository phenotypeTermRepository;
    private final GeneRepository geneRepository;




    @Autowired
    public PatientService(PatientRepository patientRepository, DiseaseRepository diseaseRepository, MedicalCenterRepository medicalCenterRepository, VariantRepository variantRepository, ClinicianRepository clinicianRepository, PhenotypeTermRepository phenotypeTermRepository, GeneRepository geneRepository) {

        this.patientRepository = patientRepository;
        this.diseaseRepository = diseaseRepository;
        this.medicalCenterRepository = medicalCenterRepository;
        this.variantRepository = variantRepository;
        this.clinicianRepository = clinicianRepository;
        this.phenotypeTermRepository = phenotypeTermRepository;
        this.geneRepository = geneRepository;
    }

    public String addPatient(Patient patient) {
        if ( patient.getDisease() != null ) {
            Long diseaseId = patient.getDisease().getId();
            patient.setDisease(diseaseRepository.findById(diseaseId).orElse(null));
        }

        if ( patient.getMedicalCenter() == null ) {
            // return an error
            return "Medical Center is required";
        }
        Long medicalCenterId = patient.getMedicalCenter().getId();
        patient.setMedicalCenter(medicalCenterRepository.findById(medicalCenterId).orElse(null));

        patientRepository.save(patient);
        return "Patient added successfully";
    }

    public List<Patient> getAllPatients() {
        return patientRepository.findAll();
    }

    public Patient getPatientById(Long id) {
        return patientRepository.findById(id).orElse(null);
    }

    public List<Patient> getPatientsByDiseaseName(String diseaseName) {
        return patientRepository.findByDiseaseName(diseaseName);
    }

    public String addDiseaseToPatient(Long patientId, Long diseaseId) {
        Disease disease = diseaseRepository.findById(diseaseId).orElse(null);
        Patient patient = patientRepository.findById(patientId).orElse(null);

        if ( disease == null ) {
            return "Disease not found";
        }

        if ( patient == null ) {
            return "Patient not found";
        }

        // if patient already has a disease, return an error
        if ( patient.getDisease() != null ) {
            return "Patient already has a disease";
        }

        patient.setDisease(disease);
        patientRepository.save(patient);
        return "Disease added to patient successfully";
    }

    public List<Patient> getPatientsByMedicalCenterId(Long medicalCenterId) {
        return patientRepository.findByMedicalCenterId(medicalCenterId);
    }

    public String addVariantToPatient(Long patientId, Long variantId) {
        // check if variant exists
        Variant variant = variantRepository.findById(variantId).orElse(null);
        Patient patient = patientRepository.findById(patientId).orElse(null);

        if ( variant == null ) {
            return "Variant not found";
        }

        // check if patient exists
        if ( patient == null ) {
            return "Patient not found";
        }

        // add the variant to the list of the patient if list is not empty, otherwise create a new list
        if ( patient.getVariants() != null ) {
            patient.getVariants().add(variant);
        } else {
            patient.setVariants(List.of(variant));
        }

        patientRepository.save(patient);
        return "Variant added to patient successfully";
    }

    public String addPatientToClinician(Patient patient, Long clinicianId) {
        if ( patient.getDisease() != null ) {
            Long diseaseId = patient.getDisease().getId();
            patient.setDisease(diseaseRepository.findById(diseaseId).orElse(null));
        }

        if ( patient.getMedicalCenter() == null ) {
            // return an error
            return "Medical Center is required";
        }
        Long medicalCenterId = patient.getMedicalCenter().getId();
        MedicalCenter medicalCenter = medicalCenterRepository.findById(medicalCenterId).orElse(null);

        if ( medicalCenter == null ) {
            return "Medical Center with id " + medicalCenterId + " does not exist";
        }

        Clinician clinician = clinicianRepository.findById(clinicianId).orElse(null);

        if ( clinician == null ) {
            return "Clinician with id " + clinicianId + " does not exist";
        }

        // add the patient to the list of the clinician if list is not empty, otherwise create a new list
        if ( clinician.getPatients() != null ) {
            clinician.getPatients().add(patient);
        } else {
            clinician.setPatients(List.of(patient));
        }

        clinicianRepository.save(clinician);
        patient.setMedicalCenter(medicalCenter);
        patientRepository.save(patient);
        return "Patient added successfully";
    }

    public List<Patient> getPatientsByClinicianId(Long clinicianId) {
        Clinician clinician = clinicianRepository.findById(clinicianId).orElse(null);

        if ( clinician == null ) {
            return null;
        }

        return clinician.getPatients();
    }

    public String addPhenotypeTermToPatient(Long patientId, Long phenotypeTermId) {
        PhenotypeTerm phenotypeTerm = phenotypeTermRepository.findById(phenotypeTermId).orElse(null);
        Patient patient = patientRepository.findById(patientId).orElse(null);

        if ( phenotypeTerm == null ) {
            return "Phenotype term with id " + phenotypeTermId + " does not exist";
        }

        if ( patient == null ) {
            return "Patient with id " + patientId + " does not exist";
        }

        // add the phenotypeTerm to the list of the patient if list is not empty, otherwise create a new list
        if ( patient.getPhenotypeTerms() != null ) {
            patient.getPhenotypeTerms().add(phenotypeTerm);
        } else {
            patient.setPhenotypeTerms(List.of(phenotypeTerm));
        }

        patientRepository.save(patient);
        return "Phenotype term added to patient successfully";
    }

    public String addGeneToPatient(Long patientId, Long geneId) {
        Gene gene = geneRepository.findById(geneId).orElse(null);
        Patient patient = patientRepository.findById(patientId).orElse(null);

        if ( gene == null ) {
            return "Gene with id " + geneId + " does not exist";
        }

        if ( patient == null ) {
            return "Patient with id " + patientId + " does not exist";
        }

        // add the gene to the list of the patient if list is not empty, otherwise create a new list
        if ( patient.getGenes() != null ) {
            patient.getGenes().add(gene);
        } else {
            patient.setGenes(List.of(gene));
        }

        patientRepository.save(patient);
        return "Gene added to patient successfully";
    }

    public String deletePhenotypeTermFromPatient(Long patientId, Long phenotypeTermId) {
        PhenotypeTerm phenotypeTerm = phenotypeTermRepository.findById(phenotypeTermId).orElse(null);
        Patient patient = patientRepository.findById(patientId).orElse(null);

        if ( phenotypeTerm == null ) {
            return "Phenotype term with id " + phenotypeTermId + " does not exist";
        }

        if ( patient == null ) {
            return "Patient with id " + patientId + " does not exist";
        }

        // if the list is null or doesn't contain the phenotypeTerm, return an error
        if ( patient.getPhenotypeTerms() == null || !patient.getPhenotypeTerms().contains(phenotypeTerm) ) {
            return "Patient does not have the phenotype term with id " + phenotypeTermId;
        }
        else {
            patient.getPhenotypeTerms().remove(phenotypeTerm);
        }

        patientRepository.save(patient);
        return "Phenotype term deleted from patient successfully";
    }

    // vectorized patient's phenotype by updating the phenotypeVector field
    public String vectorizePatientPhenotype(Long patientId) {
        //get patient from database
        Patient patient = getPatientById(patientId);

        if ( patient == null ) {
            return "Patient with id " + patientId + " does not exist";
        }

        //get patient's phenotype terms
        List<PhenotypeTerm> phenotypeTerms = patient.getPhenotypeTerms();

        if ( phenotypeTerms == null ) {
            return "Patient does not have any phenotype terms";
        }

        //get all phenotype terms ids from SsortedHPOids.txt file
        //open file under resources

        //arraylist of all phenotype term ids
        List<Integer> allPhenotypeTermIds = new ArrayList<>();

        ClassPathResource resource = new ClassPathResource("sortedHPOids.txt");
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(resource.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                //add the phenotype term id as long to the arraylist
                allPhenotypeTermIds.add(Integer.parseInt(line));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }



        //sort them by id (already sorted)
        //allPhenotypeTerms.sort((PhenotypeTerm p1, PhenotypeTerm p2) -> p1.getId().compareTo(p2.getId()));
        //provide index table for all phenotype terms
        //create a map of phenotype terms and their indices in the allPhenotypeTerms list
        Map<Integer, Integer> phenotypeTermIndexMap = new HashMap<>();

        for (int i = 0; i < allPhenotypeTermIds.size(); i++) {
            phenotypeTermIndexMap.put(allPhenotypeTermIds.get(i), i);

            //if 66
            if( allPhenotypeTermIds.get(i) == 66){
                System.out.println("yess66ss");
                System.out.println(i);
            }


        }

        //vectorize the patient's phenotype
        //create a float array of zeros of size allPhenotypeTerms.size()
        float phenotypeVector[] = new float[allPhenotypeTermIds.size()];
        for (int i = 0; i < phenotypeVector.length; i++) {
            phenotypeVector[i] = 0;
        }

        //for every phenotype term in patient's phenotype terms,
        // set the corresponding index and all its ancestors in the phenotypeVector to 1

        for (PhenotypeTerm phenotypeTerm : phenotypeTerms) {
            //convert phenotypeTerm.getId() to integer and get the index from the map
            Integer id2 = phenotypeTerm.getId().intValue();

            //print type of id2
            System.out.println( "Name " +    id2.getClass().getName());

            int index = phenotypeTermIndexMap.get( id2 );
            //set the index and all its ancestors in the phenotypeVector to 1

            // dynamic initialization of index arraylist of ancestors
            List<Integer> indexes = new ArrayList<>();
            indexes.add(index);

            while (indexes.size() > 0) {
                //get the last index from the indexes list
                int currentIndex = indexes.get(indexes.size() - 1);

                if (phenotypeVector[currentIndex] == 1) {
                    //if the index in the phenotypeVector is already 1, remove the last index from the indexes list
                    indexes.remove(indexes.size() - 1);
                    continue;
                }

                //set the index in the phenotypeVector to 1
                phenotypeVector[currentIndex] = 1;
                //remove the last index from the indexes list
                indexes.remove(indexes.size() - 1);

                //get the phenotype term from the allPhenotypeTerms list at the currentIndex
                PhenotypeTerm currentPhenotypeTerm = phenotypeTermRepository.findById((long) allPhenotypeTermIds.get(currentIndex)).orElse(null);
                //get the parents of the current phenotype term
                List<PhenotypeTerm> parents = currentPhenotypeTerm.getParents();

                //if the current phenotype term has parents, add their indices to the indexes list
                if (parents != null) {
                    for (PhenotypeTerm parent : parents) {
                        Integer pID = parent.getId().intValue();
                        int pIndex = phenotypeTermIndexMap.get( pID );
                        indexes.add(phenotypeTermIndexMap.get(pIndex));
                    }
                }
            }
        }
        //update the patient's phenotypeVector field
        patient.setPhenotypeVector(phenotypeVector);

        // save and flush the patient
        patientRepository.save(patient);

        return "Patient's phenotype vectorized successfully";
    }


    public ResponseEntity<String> addPatientWithPhenotype(PatientWithPhenotype patientDto) {
        Patient patient = patientDto.getPatient();
        List<PhenotypeTerm> phenotypeTerms = patientDto.getPhenotypeTerms();

        if ( patient.getMedicalCenter() == null ) {
            return new ResponseEntity<>("Medical Center is required", org.springframework.http.HttpStatus.BAD_REQUEST);
        }

        Long medicalCenterId = patient.getMedicalCenter().getId();
        patient.setMedicalCenter(medicalCenterRepository.findById(medicalCenterId).orElse(null));

        List<PhenotypeTerm> newPhenotypeTerms = new ArrayList<>();
        if ( phenotypeTerms != null ) {
            for ( PhenotypeTerm phenotypeTerm : phenotypeTerms ) {
                Long phenotypeTermId = phenotypeTerm.getId();
                newPhenotypeTerms.add(phenotypeTermRepository.findById(phenotypeTermId).orElse(null));
            }
        }

        patient.setPhenotypeTerms(newPhenotypeTerms);
        patientRepository.save(patient);
        return new ResponseEntity<>("Patient added successfully", org.springframework.http.HttpStatus.OK);
    }
}
