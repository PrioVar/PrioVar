package com.bio.priovar.services;

import com.bio.priovar.models.MedicalCenter;
import com.bio.priovar.models.Subscription;
import com.bio.priovar.repositories.MedicalCenterRepository;
import org.springframework.beans.factory.annotation.Autowired;
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

    public String addSubscriptionToMedicalCenter(Long medicalCenterId, Long subscriptionId) {
        // 1 indicates Basic Subscription
        // 2 indicates Premium Subscription
        // 3 indicates Enterprise Subscription
        MedicalCenter medicalCenter = medicalCenterRepository.findById(medicalCenterId).orElse(null);
        if ( medicalCenter == null ) {
            return "Medical Center with id " + medicalCenterId + " does not exist";
        }

        Subscription subscription;
        if ( subscriptionId == 1 ) {
            subscription = Subscription.BASIC;
        } else if ( subscriptionId == 2 ) {
            subscription = Subscription.PREMIUM;
        } else if ( subscriptionId == 3 ) {
            subscription = Subscription.ENTERPRISE;
        } else {
            return "Invalid subscription id";
        }

        medicalCenter.setSubscription(subscription);
        medicalCenter.setRemainingAnalyses(subscription.getAnalyses() + medicalCenter.getRemainingAnalyses());
        medicalCenterRepository.save(medicalCenter);
        return "Subscription added successfully";
    }
}
