package com.bio.priovar.repositories;

import com.bio.priovar.models.Patient;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface PatientRepository extends JpaRepository<Patient, Long> {
    List<Patient> findAllByMedicalCenter_ID(Long medicalCenterId);
}
