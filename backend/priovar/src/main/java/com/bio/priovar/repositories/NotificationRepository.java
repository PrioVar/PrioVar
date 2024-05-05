package com.bio.priovar.repositories;

import com.bio.priovar.models.Notification;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.neo4j.repository.query.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface NotificationRepository extends Neo4jRepository<Notification, Long> {

    //@Query("MATCH (n:Notification)-[:NOTIFIED_TO]->(a) WHERE ID(a) = $actorId RETURN n")
    List<Notification> findByReceiverId(Long actorId);

    List<Notification> findByReceiverIdAndIsReadFalse(Long actorId);

}
