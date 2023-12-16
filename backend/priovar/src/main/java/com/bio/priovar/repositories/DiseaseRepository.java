package com.bio.priovar.repositories;

import com.bio.priovar.models.Disease;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface DiseaseRepository extends Neo4jRepository<Disease, Long> {
    Disease findByDiseaseName(String s);
}
