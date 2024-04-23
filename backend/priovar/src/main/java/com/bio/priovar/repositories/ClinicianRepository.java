package com.bio.priovar.repositories;

import com.bio.priovar.models.Clinician;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface ClinicianRepository extends Neo4jRepository<Clinician, Long> {
    Clinician findByEmail(String email);

    List<Clinician> findAllByMedicalCenterId(Long medicalCenterId);
    Optional<Clinician> findByVcfFilesId(Long vcfFileId);
}
