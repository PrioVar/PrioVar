package com.bio.priovar.services;

import com.bio.priovar.models.MedicalCenter;
import com.bio.priovar.models.Notification;
import com.bio.priovar.repositories.MedicalCenterRepository;
import com.bio.priovar.repositories.NotificationRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class NotificationService {
    private final NotificationRepository notificationRepository;
    private final MedicalCenterRepository medicalCenterRepository;

    @Autowired
    public NotificationService(NotificationRepository notificationRepository, MedicalCenterRepository medicalCenterRepository) {
        this.notificationRepository = notificationRepository;
        this.medicalCenterRepository = medicalCenterRepository;
    }

    public Notification getNotificationById(Long id) {
        return notificationRepository.findById(id).orElse(null);
    }

    public String addNotification(Notification notification) {
        MedicalCenter medicalCenter = notification.getMedicalCenter();

        if ( medicalCenter == null ) {
            return "Medical Center is required";
        }

        Long medicalCenterId = medicalCenter.getId();
        medicalCenter = medicalCenterRepository.findById(medicalCenterId).orElse(null);

        if ( medicalCenter == null ) {
            return "Medical Center with id " + medicalCenterId + " does not exist";
        }

        notification.setMedicalCenter(medicalCenter);
        notificationRepository.save(notification);
        return "Notification added successfully";
    }

    public List<Notification> getNotificationsByMedicalCenterId(Long medicalCenterId) {
        return notificationRepository.findByMedicalCenterId(medicalCenterId);
    }

    public String markNotificationAsReadByID(Long notificationId) {
        Notification notification = notificationRepository.findById(notificationId).orElse(null);

        if ( notification == null ) {
            return "Notification with id " + notificationId + " does not exist";
        }

        notification.setIsRead(true);
        notificationRepository.save(notification);
        return "Notification marked as read";
    }

    public String deleteNotificationByID(Long notificationId) {
        Notification notification = notificationRepository.findById(notificationId).orElse(null);

        if ( notification == null ) {
            return "Notification with id " + notificationId + " does not exist";
        }

        notificationRepository.deleteById(notificationId);
        return "Notification deleted successfully";
    }

    public List<Notification> getUnreadNotificationsByMedicalCenterId(Long medicalCenterId) {
        return notificationRepository.findByMedicalCenterIdAndIsReadFalse(medicalCenterId);
    }
}
