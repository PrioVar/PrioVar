package com.bio.priovar.repositories;

import com.bio.priovar.models.Clinician;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface ClinicianRepository extends Neo4jRepository<Clinician, Long> {
    Clinician findByEmail(String email);
}
