package com.bio.priovar.repositories;

import com.bio.priovar.models.InformationRequest;
import com.bio.priovar.models.MedicalCenter;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.neo4j.repository.query.Query;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface InformationRequestRepository extends Neo4jRepository<InformationRequest, Long>{

    public List<InformationRequest> findAllByPatientMedicalCenterAndIsPending(MedicalCenter medicalCenter, boolean isPending);

    public List<InformationRequest> findAllByClinicianIdAndIsPending(Long clinicianId, boolean isPending);
}
