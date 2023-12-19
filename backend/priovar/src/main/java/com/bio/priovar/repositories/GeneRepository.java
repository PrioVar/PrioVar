package com.bio.priovar.repositories;

import com.bio.priovar.models.Gene;
import com.bio.priovar.models.PhenotypeTerm;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.neo4j.repository.query.Query;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository

public interface GeneRepository extends Neo4jRepository<Gene, Long> {

    Optional<Gene> findByGeneSymbol(String geneSymbol);

}
