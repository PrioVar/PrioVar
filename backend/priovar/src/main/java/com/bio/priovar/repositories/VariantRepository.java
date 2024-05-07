package com.bio.priovar.repositories;

import com.bio.priovar.models.Variant;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.neo4j.repository.query.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface VariantRepository extends Neo4jRepository<Variant, Long> {

    //query for: findvariant by patient id
    // MATCH (p:Patient)-[:HAS_VARIANT]->(v:Variant)
    //WHERE ID(p) = $patient_id
    //RETURN v

    @Query("MATCH (p:Patient)-[:HAS_VARIANT]->(v:Variant) WHERE ID(p) = $patient_id RETURN v")
    List<Variant> getVariantsByPatientId(Long patient_id);


}
