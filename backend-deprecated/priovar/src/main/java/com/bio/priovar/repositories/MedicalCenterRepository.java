package com.bio.priovar.repositories;

import com.bio.priovar.models.MedicalCenter;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface MedicalCenterRepository extends JpaRepository<MedicalCenter, Long> {
    List<MedicalCenter> findAll();

    Optional<MedicalCenter> findByName(String name);
}
