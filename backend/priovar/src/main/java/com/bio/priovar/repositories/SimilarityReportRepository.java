package com.bio.priovar.repositories;

import com.bio.priovar.models.SimilarityReport;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.neo4j.repository.query.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface SimilarityReportRepository extends Neo4jRepository<SimilarityReport, Long> {


    // find a similarity report by its ID
    @Query("MATCH (p:SimilarityReport) WHERE p.id = $id RETURN p")
    SimilarityReport findSimilarityReportById(Long id);


    List<SimilarityReport> findAllByPrimaryPatientId(Long id);
}