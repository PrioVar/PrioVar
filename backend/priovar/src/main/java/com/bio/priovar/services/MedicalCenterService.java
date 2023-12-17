package com.bio.priovar.services;

import com.bio.priovar.models.MedicalCenter;
import com.bio.priovar.models.Subscription;
import com.bio.priovar.models.dto.LoginObject;
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

    public ResponseEntity<LoginObject> loginMedicalCenter(String email, String password) {
        MedicalCenter medicalCenter = medicalCenterRepository.findByEmail(email);
        LoginObject loginObject = new LoginObject();
        if ( medicalCenter == null ) {
            loginObject.setMessage("Medical Center with email " + email + " does not exist");
            loginObject.setId(-1L);
            return ResponseEntity.badRequest().body(loginObject);
        }

        if ( !medicalCenter.getPassword().equals(password) ) {
            loginObject.setMessage("Invalid password");
            loginObject.setId(-1L);
            return ResponseEntity.badRequest().body(loginObject);
        }

        loginObject.setMessage("Login successful");
        loginObject.setId(medicalCenter.getId());
        return ResponseEntity.ok(loginObject);
    }

    public ResponseEntity<String> changePasswordByEmailMedicalCenter(String email, String newPass, String oldPass) {
        MedicalCenter medicalCenter = medicalCenterRepository.findByEmail(email);
        if ( medicalCenter == null ) {
            return ResponseEntity.badRequest().body("Medical Center with email " + email + " does not exist");
        }

        if ( !medicalCenter.getPassword().equals(oldPass) ) {
            return ResponseEntity.badRequest().body("Invalid password");
        }

        if ( newPass.equals(oldPass) ) {
            return ResponseEntity.badRequest().body("New password cannot be the same as old password");
        }

        medicalCenter.setPassword(newPass);
        medicalCenterRepository.save(medicalCenter);
        return ResponseEntity.ok("Password changed successfully");
    }
}
