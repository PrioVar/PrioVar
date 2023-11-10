package com.bio.priovar.repositories;

import com.bio.priovar.models.Variant;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface VariantRepository extends JpaRepository<Variant, Long> {
    List<Variant> findAllByPatient_ID(Long patientId);
}
