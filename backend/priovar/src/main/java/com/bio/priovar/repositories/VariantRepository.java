package com.bio.priovar.repositories;

import com.bio.priovar.models.Variant;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface VariantRepository extends Neo4jRepository<Variant, Long> {
}
