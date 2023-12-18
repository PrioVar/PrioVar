package com.bio.priovar.repositories;

import com.bio.priovar.models.Patient;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.neo4j.repository.query.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface PatientRepository extends Neo4jRepository<Patient, Long> {
    @Query("MATCH (p:Patient)-[:HAS_DISEASE]->(d:Disease) WHERE d.name = $diseaseName RETURN p")
    List<Patient> findByDiseaseName(String diseaseName);

    List<Patient> findByMedicalCenterId(Long medicalCenterId);

    @Query(
    "MATCH (p:Patient) "+
    "WHERE p.sex = $sexQuery " +
    "AND p.age >= $ageIntervalStartQuery "+
    "AND p.age <= $ageIntervalEndQuery "+
    "AND (size($genesQuery) = 0 OR ALL(g IN $genesQuery WHERE EXISTS((p)-[:ASSOCIATED_WITH_GENE]->(:Gene) WHERE Gene.geneSymbol = g))) "+
    "AND (size($phenotypeTermsQuery) = 0 OR ALL(pt IN $phenotypeTermsQuery WHERE EXISTS((p)-[:HAS_PHENOTYPE_TERM]->(:PhenotypeTerm) WHERE PhenotypeTerm.name = pt))) "+
    "RETURN p")
    List<Patient> findPatientsBySexAndAgeIntervalAndGenesAndPhenotypeTerms( String sexQuery, int ageIntervalStartQuery, int ageIntervalEndQuery, List<String> genesQuery, List<String> phenotypeTermsQuery);
}
