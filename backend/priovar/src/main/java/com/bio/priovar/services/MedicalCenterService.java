package com.bio.priovar.services;

import com.bio.priovar.models.MedicalCenter;
import com.bio.priovar.repositories.MedicalCenterRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class MedicalCenterService {

    private final MedicalCenterRepository medicalCenterRepository;

    @Autowired
    public MedicalCenterService(MedicalCenterRepository medicalCenterRepository) {
        this.medicalCenterRepository = medicalCenterRepository;
    }

    public List<MedicalCenter> getAllMedicalCenters() {
        return medicalCenterRepository.findAll();
    }

    public void addMedicalCenter(MedicalCenter medicalCenter) {
        // check if the name of the medical center already exists
        Optional<MedicalCenter> medicalCenterOptional = medicalCenterRepository.findByName(medicalCenter.getName());

        if ( medicalCenterOptional.isPresent() ) {
            throw new IllegalStateException("Medical center with name: " + medicalCenter.getName() + " already exists!");
        }

        medicalCenterRepository.save(medicalCenter);
    }
}
