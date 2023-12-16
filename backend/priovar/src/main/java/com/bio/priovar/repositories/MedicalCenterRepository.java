package com.bio.priovar.repositories;

import com.bio.priovar.models.MedicalCenter;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface MedicalCenterRepository extends Neo4jRepository<MedicalCenter, Long> {
    MedicalCenter findByEmail(String email);
}
