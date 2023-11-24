package com.bio.priovar.services;

import com.bio.priovar.models.MedicalCenter;
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
}
