package com.bio.priovar.repositories;

import com.bio.priovar.models.Admin;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface AdminRepository extends Neo4jRepository<Admin, Long> {
    Admin findAdminByEmail(String email);
}
