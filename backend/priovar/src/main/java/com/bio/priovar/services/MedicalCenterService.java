package com.bio.priovar.services;

import com.bio.priovar.models.MedicalCenter;
import com.bio.priovar.models.Subscription;
import com.bio.priovar.repositories.MedicalCenterRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class MedicalCenterService {
    private final MedicalCenterRepository medicalCenterRepository;

    @Autowired
    public MedicalCenterService(MedicalCenterRepository medicalCenterRepository) {
        this.medicalCenterRepository = medicalCenterRepository;
    }

    public void addMedicalCenter(MedicalCenter medicalCenter) {
        medicalCenterRepository.save(medicalCenter);
    }

    public List<MedicalCenter> getAllMedicalCenters() {
        return medicalCenterRepository.findAll();
    }

    public MedicalCenter getMedicalCenterById(Long medicalCenterId) {
        return medicalCenterRepository.findById(medicalCenterId).orElse(null);
    }

    public ResponseEntity<String> addSubscriptionToMedicalCenter(Long medicalCenterId, Long subscriptionId) {
        // 1 indicates Basic Subscription
        // 2 indicates Premium Subscription
        // 3 indicates Enterprise Subscription
        MedicalCenter medicalCenter = medicalCenterRepository.findById(medicalCenterId).orElse(null);
        if ( medicalCenter == null ) {
            return ResponseEntity.badRequest().body("Medical Center with id " + medicalCenterId + " does not exist");
        }

        Subscription subscription;
        if ( subscriptionId == 1 ) {
            subscription = Subscription.BASIC;
        } else if ( subscriptionId == 2 ) {
            subscription = Subscription.PREMIUM;
        } else if ( subscriptionId == 3 ) {
            subscription = Subscription.ENTERPRISE;
        } else {
            return ResponseEntity.badRequest().body("Invalid subscription id");
        }

        medicalCenter.setSubscription(subscription);
        medicalCenter.setRemainingAnalyses(subscription.getAnalyses() + medicalCenter.getRemainingAnalyses());
        medicalCenterRepository.save(medicalCenter);
        return ResponseEntity.ok("Subscription added successfully");
    }

    public ResponseEntity<String> loginMedicalCenter(String email, String password) {
        MedicalCenter medicalCenter = medicalCenterRepository.findByEmail(email);
        if ( medicalCenter == null ) {
            return ResponseEntity.badRequest().body("Medical Center with email " + email + " does not exist");
        }

        if ( !medicalCenter.getPassword().equals(password) ) {
            return ResponseEntity.badRequest().body("Invalid password");
        }

        return ResponseEntity.ok("Login successful");
    }
}
