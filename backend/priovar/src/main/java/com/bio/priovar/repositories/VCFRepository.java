package com.bio.priovar.repositories;

import com.bio.priovar.models.VCFFile;

import java.util.List;

import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface VCFRepository extends Neo4jRepository<VCFFile, Long> {    
    List<VCFFile> findAllByMedicalCenterId(Long medicalCenterId);
}

