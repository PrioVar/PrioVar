package com.bio.priovar.controllers;

import com.bio.priovar.models.Admin;
import com.bio.priovar.models.Clinician;
import com.bio.priovar.models.MedicalCenter;
import com.bio.priovar.models.Subscription;
import com.bio.priovar.repositories.AdminRepository;
import com.bio.priovar.repositories.ClinicianRepository;
import com.bio.priovar.repositories.MedicalCenterRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/initialize")
@CrossOrigin
public class InitializerController {
    private final MedicalCenterRepository medicalCenterRepository;
    private final ClinicianRepository clinicianRepository;
    private final AdminRepository adminRepository;

    @Autowired
    public InitializerController(MedicalCenterRepository medicalCenterRepository, ClinicianRepository clinicianRepository, AdminRepository adminRepository) {
        this.medicalCenterRepository = medicalCenterRepository;
        this.clinicianRepository = clinicianRepository;
        this.adminRepository = adminRepository;
    }

    @PostMapping()
    public ResponseEntity<String> initialize() {
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
