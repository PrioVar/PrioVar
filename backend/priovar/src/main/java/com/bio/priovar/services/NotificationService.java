package com.bio.priovar.services;

import com.bio.priovar.models.Notification;
import com.bio.priovar.repositories.MedicalCenterRepository;
import com.bio.priovar.repositories.NotificationRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class NotificationService {
    private final NotificationRepository notificationRepository;

    @Autowired
    public NotificationService(NotificationRepository notificationRepository, MedicalCenterRepository medicalCenterRepository) {
        this.notificationRepository = notificationRepository;
    }

    public Notification getNotificationById(Long id) {
        return notificationRepository.findById(id).orElse(null);
    }


    public List<Notification> getNotificationsByActorId(Long actorId) {
        return notificationRepository.findByReceiverId(actorId);
    }

    public ResponseEntity<String> markNotificationAsReadByID(Long notificationId) {
        Notification notification = notificationRepository.findById(notificationId).orElse(null);

        if ( notification == null ) {
            return ResponseEntity.ok("Notification with id " + notificationId + " does not exist");
        }

        notification.setIsRead(true);
        notificationRepository.save(notification);
        return ResponseEntity.ok("Notification marked as read");
    }

    public String deleteNotificationByID(Long notificationId) {
        Notification notification = notificationRepository.findById(notificationId).orElse(null);

        if ( notification == null ) {
            return "Notification with id " + notificationId + " does not exist";
        }

        notificationRepository.deleteById(notificationId);
        return "Notification deleted successfully";
    }

    public ResponseEntity<String> markAllNotificationsAsReadByActorId(Long actorId) {
        List<Notification> notifications = notificationRepository.findByReceiverId(actorId);

        for (Notification notification : notifications) {
            notification.setIsRead(true);
            notificationRepository.save(notification);
        }

        return ResponseEntity.ok("All notifications marked as read");
    }

    public List<Notification> getUnreadNotificationsByActorId(Long actorId) {
        return notificationRepository.findByReceiverIdAndIsReadFalse(actorId);
    }
}
