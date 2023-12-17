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

    // find all similarity reports by its primary patient ID
    @Query("MATCH (p:SimilarityReport) WHERE p.primaryPatient.id = $id RETURN p")
    List<SimilarityReport> findAllSimilarityReportsByPrimaryPatientId(Long id);


    // find all similarity reports by patient ID
    @Query("MATCH (p:SimilarityReport) WHERE p.primaryPatient.id = $id OR p.secondaryPatient.id = $id RETURN p")
    List<SimilarityReport> findAllSimilarityReportsByPatientId(Long id);


}