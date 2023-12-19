package com.bio.priovar.controllers;

import com.bio.priovar.models.*;
import com.bio.priovar.repositories.*;
import com.bio.priovar.services.GraphLoaderService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.ArrayList;
import java.util.List;

@RestController
@RequestMapping("/initialize")
@CrossOrigin
public class InitializerController {
    private final MedicalCenterRepository medicalCenterRepository;
    private final ClinicianRepository clinicianRepository;
    private final AdminRepository adminRepository;
    private final GraphLoaderService graphLoaderService;
    private final PatientRepository patientRepository;
    private final PhenotypeTermRepository phenotypeTermRepository;
    private final DiseaseRepository diseaseRepository;
    private final GeneRepository geneRepository;

    @Autowired
    public InitializerController(MedicalCenterRepository medicalCenterRepository, ClinicianRepository clinicianRepository, AdminRepository adminRepository, GraphLoaderService graphLoaderService, PatientRepository patientRepository, PhenotypeTermRepository phenotypeTermRepository, DiseaseRepository diseaseRepository, GeneRepository geneRepository) {
        this.medicalCenterRepository = medicalCenterRepository;
        this.clinicianRepository = clinicianRepository;
        this.adminRepository = adminRepository;
        this.graphLoaderService = graphLoaderService;
        this.patientRepository = patientRepository;
        this.phenotypeTermRepository = phenotypeTermRepository;
        this.diseaseRepository = diseaseRepository;
        this.geneRepository = geneRepository;
    }

    @PostMapping()
    public ResponseEntity<String> initialize() {

        // if the length of the PhenotypeTerm table is greater than 0, then skip loading PhenotypeTerm data
        if ( phenotypeTermRepository.count() > 0) {
            System.out.println("HPO data already loaded");
        } else {
            graphLoaderService.startHPODataLoading();
            System.out.println("HPO data loaded");
            graphLoaderService.startDiseaseDataLoading();
            System.out.println("Disease data loaded");
            graphLoaderService.startGeneDatafromHPOLoading();
            System.out.println("Gene data loaded");
        }

        MedicalCenter liva = new MedicalCenter();
        liva.setName("Liva");
        liva.setAddress("Kızılay, Ankara");
        liva.setEmail("liva-mail@liva");
        liva.setPassword("123");
        liva.setPhone("05555555555");
        liva.setSubscription(Subscription.BASIC);
        liva.setRemainingAnalyses(10);

        medicalCenterRepository.save(liva);

        Clinician clinician1 = new Clinician();

        clinician1.setName("Mehmet Kılıç");
        clinician1.setEmail("mehmet.kilic@acibadem");
        clinician1.setPassword("123");
        clinician1.setMedicalCenter(liva);
        clinician1.setPatients(new ArrayList<>());
        clinicianRepository.save(clinician1);

        Patient patient1 = new Patient();
        patient1.setName("Ali Veli");
        patient1.setAge(25);
        patient1.setSex("male");
        patient1.setMedicalCenter(liva);
        Disease disease1 = diseaseRepository.findByDiseaseName("White-Kernohan syndrome");
        patient1.setDisease(disease1);
        PhenotypeTerm phenotypeTerm1 = phenotypeTermRepository.findById(7617L).get(); //from group 3
        PhenotypeTerm phenotypeTerm2 = phenotypeTermRepository.findById(271L).get(); //group 1
        PhenotypeTerm phenotypeTerm7 = phenotypeTermRepository.findById(841L).get(); //group 2
        List<PhenotypeTerm> phenotypeTerms = new ArrayList<>();
        phenotypeTerms.add(phenotypeTerm1);
        phenotypeTerms.add(phenotypeTerm2);
        phenotypeTerms.add(phenotypeTerm7);

        patient1.setPhenotypeTerms(phenotypeTerms);
        patient1.setGenes(genes1);
        patientRepository.save(patient1);

        Patient patient2 = new Patient();
        patient2.setName("Ayşe Fatma");
        patient2.setAge(40);
        patient2.setSex("female");
        PhenotypeTerm phenotypeTerm5 = phenotypeTermRepository.findById(929L).get(); //from group 1
        PhenotypeTerm phenotypeTerm6 = phenotypeTermRepository.findById(847L).get(); //from group 2
        PhenotypeTerm phenotypeTerm8 = phenotypeTermRepository.findById(1106L).get(); //from group 3
        List<PhenotypeTerm> phenotypeTerms1 = new ArrayList<>();
        phenotypeTerms1.add(phenotypeTerm5);
        phenotypeTerms1.add(phenotypeTerm6);
        phenotypeTerms1.add(phenotypeTerm8);
        patient2.setPhenotypeTerms(phenotypeTerms1);
        patient2.setMedicalCenter(liva);
        patientRepository.save(patient2);

        Patient patient3 = new Patient();
        patient3.setName("Ahmet Mehmet");
        patient3.setAge(33);
        patient3.setSex("male");
        PhenotypeTerm phenotypeTerm3 = phenotypeTermRepository.findById(40085L).get(); //from group 2
        PhenotypeTerm phenotypeTerm4 = phenotypeTermRepository.findById(19L).get();
        List<PhenotypeTerm> phenotypeTerms2 = new ArrayList<>();
        phenotypeTerms2.add(phenotypeTerm3);
        phenotypeTerms2.add(phenotypeTerm4);
        Gene gene5 = geneRepository.findById(5000L).get();
        Gene gene6 = geneRepository.findById(5001L).get();
        List<Gene> genes3 = new ArrayList<>();
        genes3.add(gene5);
        genes3.add(gene6);
        patient3.setGenes(genes3);
        patient3.setPhenotypeTerms(phenotypeTerms2);
        patient3.setMedicalCenter(liva);
        patientRepository.save(patient3);

        Patient patient4 = new Patient();
        patient4.setName("Ece Nur");
        patient4.setAge(20);
        patient4.setSex("female");
        PhenotypeTerm phenotypeTerm9 = phenotypeTermRepository.findById(3351L).get(); //from group 2
        PhenotypeTerm phenotypeTerm10 = phenotypeTermRepository.findById(5346L).get(); // from group 1
        List<PhenotypeTerm> phenotypeTerms3 = new ArrayList<>();
        phenotypeTerms3.add(phenotypeTerm9);
        phenotypeTerms3.add(phenotypeTerm10);
        patient4.setPhenotypeTerms(phenotypeTerms3);

        patient4.setMedicalCenter(liva);
        patientRepository.save(patient4);

        List<Patient> patients = clinician1.getPatients();
        patients.add(patient1);
        patients.add(patient2);
        clinician1.setPatients(patients);
        clinicianRepository.save(clinician1);

        // MEDICAL CENTER 2

        MedicalCenter acibadem = new MedicalCenter();
        acibadem.setName("Acıbadem");
        acibadem.setAddress("Acıbadem, İstanbul");
        acibadem.setEmail("acibadem-mail@acibadem");
        acibadem.setPassword("123");
        acibadem.setPhone("05555555555");
        acibadem.setSubscription(Subscription.PREMIUM);
        acibadem.setRemainingAnalyses(20);

        medicalCenterRepository.save(acibadem);

        Clinician clinician2 = new Clinician();

        clinician2.setName("Ahmet Karaca");
        clinician2.setEmail("ahmet.karaca@acibadem");
        clinician2.setPassword("123");
        clinician2.setMedicalCenter(acibadem);
        clinicianRepository.save(clinician2);

        Patient patient5 = new Patient();
        patient5.setName("Mehmet Ali");
        List<PhenotypeTerm> phenotypeTerms4 = new ArrayList<>();
        phenotypeTerms4.add(phenotypeTerm9);//g2
        patient5.setPhenotypeTerms(phenotypeTerms4);
        patient5.setAge(29);
        patient5.setSex("male");
        patient5.setMedicalCenter(acibadem);
        patientRepository.save(patient5);


        // ADMIN

        Admin admin = new Admin();
        admin.setEmail("alperen@priovar");
        admin.setPassword("123");
        adminRepository.save(admin);

        return ResponseEntity.ok("Initialized Succesfully!");
    }
}
