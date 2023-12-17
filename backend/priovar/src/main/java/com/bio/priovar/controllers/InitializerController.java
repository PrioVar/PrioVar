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

    @Autowired
    public InitializerController(MedicalCenterRepository medicalCenterRepository, ClinicianRepository clinicianRepository, AdminRepository adminRepository, GraphLoaderService graphLoaderService, PatientRepository patientRepository, PhenotypeTermRepository phenotypeTermRepository, DiseaseRepository diseaseRepository) {
        this.medicalCenterRepository = medicalCenterRepository;
        this.clinicianRepository = clinicianRepository;
        this.adminRepository = adminRepository;
        this.graphLoaderService = graphLoaderService;
        this.patientRepository = patientRepository;
        this.phenotypeTermRepository = phenotypeTermRepository;
        this.diseaseRepository = diseaseRepository;
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
        PhenotypeTerm phenotypeTerm1 = phenotypeTermRepository.findById(26L).get();
        PhenotypeTerm phenotypeTerm2 = phenotypeTermRepository.findById(25L).get();
        List<PhenotypeTerm> phenotypeTerms = new ArrayList<>();
        phenotypeTerms.add(phenotypeTerm1);
        phenotypeTerms.add(phenotypeTerm2);
        patient1.setPhenotypeTerms(phenotypeTerms);
        patientRepository.save(patient1);

        Patient patient2 = new Patient();
        patient2.setName("Ayşe Fatma");
        patient2.setAge(40);
        patient2.setSex("female");
        patient2.setMedicalCenter(liva);
        patientRepository.save(patient2);

        Patient patient3 = new Patient();
        patient3.setName("Ahmet Mehmet");
        patient3.setAge(33);
        patient3.setSex("male");
        PhenotypeTerm phenotypeTerm3 = phenotypeTermRepository.findById(9L).get();
        PhenotypeTerm phenotypeTerm4 = phenotypeTermRepository.findById(19L).get();
        List<PhenotypeTerm> phenotypeTerms2 = new ArrayList<>();
        phenotypeTerms2.add(phenotypeTerm3);
        phenotypeTerms2.add(phenotypeTerm4);
        patient3.setPhenotypeTerms(phenotypeTerms2);
        patient3.setMedicalCenter(liva);
        patientRepository.save(patient3);

        Patient patient4 = new Patient();
        patient4.setName("Ece Nur");
        patient4.setAge(20);
        patient4.setSex("female");
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

        // ADMIN

        Admin admin = new Admin();
        admin.setEmail("alperen@priovar");
        admin.setPassword("123");
        adminRepository.save(admin);

        return ResponseEntity.ok("Initialized Succesfully!");
    }
}
