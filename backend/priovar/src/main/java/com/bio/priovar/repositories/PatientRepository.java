package com.bio.priovar.repositories;

import com.bio.priovar.models.MedicalCenter;
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
    List<Patient> findByName(String name);
    Patient findByGenesusId(String genesusId);
    Patient findByVcfFileFileName(String fileName);
    Patient findByVcfFileId(Long vcfFileId);

    //Find patients clinicians healthcenter by patient id

    @Query(
            "MATCH (p:Patient) " +
                    "WHERE ($sexQuery IS NULL OR p.sex = $sexQuery) " +
                    "AND ($ageIntervalStartQuery IS NULL OR p.age >= $ageIntervalStartQuery) " +
                    "AND ($ageIntervalEndQuery IS NULL OR p.age <= $ageIntervalEndQuery) " +
                    "AND (size($genesQuery) = 0 OR $genesQuery IS NULL OR ALL(g IN $genesQuery WHERE (p)-[:HAS_GENE]->(:Gene {geneSymbol: g}))) " +
                    "AND (size($phenotypeTermsQuery) = 0 OR $phenotypeTermsQuery IS NULL OR ALL(pt IN $phenotypeTermsQuery WHERE (p)-[:HAS_PHENOTYPE_TERM]->(:PhenotypeTerm {id: pt}))) " +
                    "RETURN p")
    List<Patient> findPatientsBySexAndAgeIntervalAndGenesAndPhenotypeTerms( String sexQuery, int ageIntervalStartQuery, int ageIntervalEndQuery, List<String> genesQuery, List<Long> phenotypeTermsQuery);
}
