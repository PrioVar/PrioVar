package com.bio.priovar.repositories;

import com.bio.priovar.models.ClinicianComment;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface ClinicianCommentRepository extends Neo4jRepository<ClinicianComment, Long> {
}
