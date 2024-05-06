package com.bio.priovar.services;

import com.bio.priovar.models.*;
import com.bio.priovar.models.dto.PatientDTO;
import com.bio.priovar.models.dto.PatientWithPhenotype;
import com.bio.priovar.models.dto.VCFFileDTO;
import com.bio.priovar.repositories.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ClassPathResource;
import org.springframework.http.HttpStatus;
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
    private final VCFRepository vcfRepository;
    private final ClinicianRepository clinicanRepository;




    @Autowired
    public PatientService(PatientRepository patientRepository, 
                        DiseaseRepository diseaseRepository, 
                        MedicalCenterRepository medicalCenterRepository, 
                        VariantRepository variantRepository, 
                        ClinicianRepository clinicianRepository, 
                        PhenotypeTermRepository phenotypeTermRepository, 
                        GeneRepository geneRepository, 
                        VCFRepository vcfRepository, ClinicianRepository clinicanRepository) {


        this.patientRepository = patientRepository;
        this.diseaseRepository = diseaseRepository;
        this.medicalCenterRepository = medicalCenterRepository;
        this.variantRepository = variantRepository;
        this.clinicianRepository = clinicianRepository;
        this.phenotypeTermRepository = phenotypeTermRepository;
        this.geneRepository = geneRepository;
        this.vcfRepository = vcfRepository;
        this.clinicanRepository = clinicanRepository;
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

        patient.setDisease(disease);
        patientRepository.save(patient);
        return "Disease added to patient successfully";
    }

    public List<PatientDTO> getPatientsByMedicalCenterId(Long medicalCenterId) {
        MedicalCenter medicalCenter = medicalCenterRepository.findById(medicalCenterId).orElse(null);
        
        if( medicalCenter == null ) {
            return null;
        }

        List<Patient> patients = patientRepository.findByMedicalCenterId(medicalCenterId);
        List<PatientDTO> patientDTOs = new ArrayList<>();
        for( Patient patient : patients ) {
            VCFFile vcfFile = patient.getVcfFile();
            VCFFileDTO vcfFileDTO = null;
            Long clinicianId = null;
            if( vcfFile != null) {
                Clinician clinician = clinicianRepository.findByVcfFilesId(vcfFile.getId()).orElse(null);
                String clinicianName = (clinician != null) ? clinician.getName() : "";
                clinicianId = (clinician != null) ? clinician.getId() : null;
                vcfFileDTO = new VCFFileDTO(vcfFile.getId(), 
                                            vcfFile.getFileName(), vcfFile.getClinicianComments(), 
                                            clinicianName,
                                            vcfFile.getFileStatus(),
                                            vcfFile.getCreatedAt(),
                                            vcfFile.getFinishedAt());
            }
            PatientDTO patientDTO = new PatientDTO(patient.getId(), 
                                                patient.getName(), 
                                                patient.getAge(),
                                                patient.getSex(), 
                                                vcfFileDTO,
                                                clinicianId);
            patientDTOs.add(patientDTO);
        }
        return patientDTOs;
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

    public List<PatientDTO> getPatientsByClinicianId(Long clinicianId) {
        Clinician clinician = clinicianRepository.findById(clinicianId).orElse(null);

        if ( clinician == null ) {
            return null;
        }

        List<Patient> patients = clinician.getPatients();
        List<PatientDTO> patientDTOs = new ArrayList<>();
        for( Patient patient : patients ) {
            VCFFile vcfFile = patient.getVcfFile();
            VCFFileDTO vcfFileDTO = null;
            if( vcfFile != null) {

                vcfFileDTO = new VCFFileDTO(vcfFile.getId(), 
                                            vcfFile.getFileName(), vcfFile.getClinicianComments(), 
                                            clinician.getName(),
                                            vcfFile.getFileStatus(),
                                            vcfFile.getCreatedAt(),
                                            vcfFile.getFinishedAt());
            }
            
            PatientDTO patientDTO = new PatientDTO(patient.getId(), 
                                                patient.getName(), 
                                                patient.getAge(),
                                                patient.getSex(), 
                                                vcfFileDTO,
                                                clinicianId);
            patientDTOs.add(patientDTO);
        }
        return patientDTOs;
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

                //print indexes
                //System.out.println("Indexes: " + indexes);

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
                if (parents != null && parents.size() > 0) {
                    for (PhenotypeTerm parent : parents) {
                        Integer pID = parent.getId().intValue();
                        int pIndex = phenotypeTermIndexMap.get( pID );
                        indexes.add(pIndex);
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
        Long vcfFileId = patientDto.getVcfFileId();
        Long clinicianId = patientDto.getClinicianId();
        Clinician clinician = clinicianRepository.findById(clinicianId).orElse(null);
        
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
        
        patient = patientRepository.save(patient);

        if( clinician != null){
            clinician.getPatients().add(patient);
            clinicianRepository.save(clinician);
        }
        Long patientId = patient.getId();
        setVCFFileOfPatient(patientId, vcfFileId);
        
        return new ResponseEntity<>("Patient added successfully", org.springframework.http.HttpStatus.OK);
    }

    public Patient getPatientByGenesusId(String genesusId) {
        return patientRepository.findByGenesusId(genesusId);
    }

    public List<String> getPatientByFileName(String fileName) {
        Patient patient =  patientRepository.findByVcfFileFileName(fileName);
        List<PhenotypeTerm> terms = patient.getPhenotypeTerms();
        List<String> termNames = new ArrayList<>();
        for (PhenotypeTerm term : terms) {
            termNames.add(term.getName());
        }
        return termNames;
    }

    public Patient getPatientForDetailedView() {
        Patient patient = patientRepository.findByName("Ali Veli").get(0);
        return patient;
    }

    public ResponseEntity<String> setVCFFileOfPatient(Long patientId, Long vcfFileId) {
        try {
            Patient patient = patientRepository.findById(patientId)
                    .orElseThrow(() -> new IllegalArgumentException("Patient with id " + patientId + " does not exist"));
            VCFFile vcfFile = vcfRepository.findById(vcfFileId)
                    .orElseThrow(() -> new IllegalArgumentException("VCF File with id " + vcfFileId + " does not exist"));
            vcfFile.setFileStatus(VCFFile.FileStatus.FILE_ANNOTATED);
            patient.setVcfFile(vcfFile);
            patientRepository.save(patient);
            vcfRepository.save(vcfFile);
            return ResponseEntity.ok("VCF file set successfully");
        } catch (Exception e) {
            // Log the exception or handle it as needed
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Failed to set VCF file: " + e.getMessage());
        }
        
    }

    public List<PatientDTO> getRequestedPatients(Long medicalCenterId) {
        //List<Patient> patients = clinicianRepository.findRequestedPatients(medicalCenterId);
        //first find all clinicians of a medical center
        List<Clinician> clinicians = clinicianRepository.findAllByMedicalCenterId(medicalCenterId);
        List<Patient> patients = new ArrayList<>();
        for (Clinician clinician : clinicians) {
            //if the clinician has requested patients, add them to the list
            if (clinician.getRequestedPatients() != null) {
                patients.addAll(clinician.getRequestedPatients());
            }
        }
        List<PatientDTO> patientDTOs = new ArrayList<>();
        for( Patient patient : patients ) {
            VCFFile vcfFile = patient.getVcfFile();
            VCFFileDTO vcfFileDTO = null;
            Long clinicianId = null;
            if( vcfFile != null) {
                Clinician clinician = clinicianRepository.findByVcfFilesId(vcfFile.getId()).orElse(null);
                String clinicianName = (clinician != null) ? clinician.getName() : "";
                clinicianId = (clinician != null) ? clinician.getId() : null;
                vcfFileDTO = new VCFFileDTO(vcfFile.getId(),
                                            vcfFile.getFileName(), vcfFile.getClinicianComments(),
                                            clinicianName,
                                            vcfFile.getFileStatus(),
                                            vcfFile.getCreatedAt(),
                                            vcfFile.getFinishedAt());
            }
            PatientDTO patientDTO = new PatientDTO(patient.getId(),
                                                patient.getName(),
                                                patient.getAge(),
                                                patient.getSex(),
                                                vcfFileDTO,
                                                clinicianId);
            patientDTOs.add(patientDTO);
        }
        return patientDTOs;
    }

    public List<PatientDTO> getAllAvailablePatients(Long medicalCenterId) {
     //get all patients from the medical center, from clinicians patients and clinicians requested patients
        List<PatientDTO> requestedPatients = getRequestedPatients(medicalCenterId);
        List<PatientDTO> medicalCenterPatients = getPatientsByMedicalCenterId(medicalCenterId);
        List<PatientDTO> allPatients = new ArrayList<>();
        allPatients.addAll(requestedPatients);
        allPatients.addAll(medicalCenterPatients);
        return allPatients;
     }


    public String deletePatient(Long patientId) {
        try {
            Patient patient = patientRepository.findById(patientId)
                    .orElseThrow(() -> new IllegalArgumentException("Patient with id " + patientId + " does not exist"));
            VCFFile vcfFile = patient.getVcfFile();
            patientRepository.delete(patient);
            if ( vcfFile != null ) {
                vcfRepository.delete(vcfFile);
            }
            return "Patient deleted successfully";
        } catch (Exception e) {
            // Log the exception or handle it as needed
            return "Failed to delete patient: " + e.getMessage();
        }
    }

    public List<PhenotypeTerm> getPhenotypeTermsOfPatient(Long patientId) {
        
        try {
            Patient patient = patientRepository.findById(patientId)
                    .orElseThrow(() -> new IllegalArgumentException("Patient with id " + patientId + " does not exist"));
            return patient.getPhenotypeTerms();
        } catch (Exception e) {
            // Log the exception or handle it as needed
            return null;
        }
    }

    public String deletePhenotypeTermFromPatientByPhenotypeTermId(Long patientId, Long phenotypeTermId) {
        try {
            Patient patient = patientRepository.findById(patientId)
                    .orElseThrow(() -> new IllegalArgumentException("Patient with id " + patientId + " does not exist"));
            PhenotypeTerm phenotypeTerm = phenotypeTermRepository.findById(phenotypeTermId)
                    .orElseThrow(() -> new IllegalArgumentException("Phenotype term with id " + phenotypeTermId + " does not exist"));
            patient.getPhenotypeTerms().remove(phenotypeTerm);
            patientRepository.save(patient);
            return "Phenotype term deleted from patient successfully";
        } catch (Exception e) {
            // Log the exception or handle it as needed
            return "Failed to delete phenotype term from patient: " + e.getMessage();
        }
    }

    public String addPhenotypeTermFromPatientByPhenotypeTermId(Long patientId, List<Long>  phenotypeTerms) {
        try {
            Patient patient = patientRepository.findById(patientId)
                    .orElseThrow(() -> new IllegalArgumentException("Patient with id " + patientId + " does not exist"));
            List<PhenotypeTerm> phenotypeTermList = new ArrayList<>();
            for (Long phenotypeTermId : phenotypeTerms) {
                PhenotypeTerm phenotypeTerm = phenotypeTermRepository.findById(phenotypeTermId)
                        .orElseThrow(() -> new IllegalArgumentException("Phenotype term with id " + phenotypeTermId + " does not exist"));
                phenotypeTermList.add(phenotypeTerm);
            }
            patient.getPhenotypeTerms().addAll(phenotypeTermList);
            patientRepository.save(patient);
            return "Phenotype term added to patient successfully";
        } catch (Exception e) {
            // Log the exception or handle it as needed
            return "Failed to add phenotype term to patient: " + e.getMessage();
        }
    }
}
