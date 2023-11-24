package com.bio.priovar.repositories;

import com.bio.priovar.models.Patient;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface PatientRepository extends Neo4jRepository<Patient, Long> {
    @Query("MATCH (p:Patient)-[:HAS_DISEASE]->(d:Disease) WHERE d.name = $diseaseName RETURN p")
    List<Patient> findByDiseaseName(String diseaseName);

    List<Patient> findByMedicalCenterId(Long medicalCenterId);
}
