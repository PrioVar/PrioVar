package com.bio.priovar.repositories;

import com.bio.priovar.models.Patient;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface PatientRepository extends Neo4jRepository<Patient, Long> {
}