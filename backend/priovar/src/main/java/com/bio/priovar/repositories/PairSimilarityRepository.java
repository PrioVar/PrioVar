package com.bio.priovar.repositories;

import com.bio.priovar.models.PairSimilarity;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.neo4j.repository.query.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface PairSimilarityRepository extends Neo4jRepository<PairSimilarity, Long> {
    @Query("MATCH (p:PairSimilarity) WHERE p.id = $id RETURN p")
    PairSimilarity findAllPairSimilaritesById(Long id);

    // find all similarity reports by its primary patient ID
    @Query("MATCH (p:PairSimilarity) WHERE p.primaryPatient.id = $id RETURN p")
    List<PairSimilarity> findAllPairSimilaritiesByPrimaryPatientId(Long id);


    // find all similarity reports by patient ID
    @Query("MATCH (p:PairSimilarity) WHERE p.primaryPatient.id = $id OR p.secondaryPatient.id = $id RETURN p")
    List<PairSimilarity> findAllPairSimilaritiesByPatientId(Long id);
}
