package com.bio.priovar.repositories;

import com.bio.priovar.models.Chat;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ChatRepository extends Neo4jRepository<Chat, Long> {
    List<Chat> findAllByMedicalCenterId(Long medicalCenterId);
}
