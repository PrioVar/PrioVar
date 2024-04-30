package com.bio.priovar.repositories;

import com.bio.priovar.models.GraphChat;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface GraphChatRepository extends Neo4jRepository<GraphChat, Long> {
    List<GraphChat> findAllByMedicalCenterId(Long medicalCenterId);
}
