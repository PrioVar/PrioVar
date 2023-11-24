package com.bio.priovar.repositories;

import com.bio.priovar.models.PhenotypeTerm;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface PhenotypeTermRepository extends Neo4jRepository<PhenotypeTerm, Long> {
    // find a phenotype term by its HPO ID
    @Query("MATCH (p:PhenotypeTerm) WHERE p.hpoId = $hpoId RETURN p")
    Optional<PhenotypeTerm> findPhenotypeTermByHpoId(String hpoId);
}
