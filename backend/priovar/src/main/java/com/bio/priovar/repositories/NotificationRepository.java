package com.bio.priovar.repositories;

import com.bio.priovar.models.Notification;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface NotificationRepository extends Neo4jRepository<Notification, Long> {
    List<Notification> findByReceiverId(Long actorId);
    List<Notification> findByReceiverIdAndIsReadFalse(Long actorId);

}
