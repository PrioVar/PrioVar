package com.bio.priovar.controllers;

import com.bio.priovar.models.Notification;
import com.bio.priovar.services.NotificationService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/notification")
@CrossOrigin
public class NotificationController {
    private final NotificationService notificationService;

    @Autowired
    public NotificationController(NotificationService notificationService) {
        this.notificationService = notificationService;
    }

    @GetMapping("/{notificationId}")
    public Notification getNotificationById(@PathVariable("notificationId") Long id) {
        return notificationService.getNotificationById(id);
    }

    @GetMapping("/byMedicalCenter/{medicalCenterId}")
    public List<Notification> getNotificationsByMedicalCenterId(@PathVariable("medicalCenterId") Long medicalCenterId) {
        return notificationService.getNotificationsByMedicalCenterId(medicalCenterId);
    }

    @GetMapping("/byMedicalCenter/{medicalCenterId}/unread")
    public List<Notification> getUnreadNotificationsByMedicalCenterId(@PathVariable("medicalCenterId") Long medicalCenterId) {
        return notificationService.getUnreadNotificationsByMedicalCenterId(medicalCenterId);
    }

    @PostMapping("/add")
    public ResponseEntity<String> addNotification(@RequestBody Notification notification) {
        return new ResponseEntity<>(notificationService.addNotification(notification), notification.getMedicalCenter() == null ? org.springframework.http.HttpStatus.BAD_REQUEST : org.springframework.http.HttpStatus.OK);
    }

    @PostMapping("/markRead/{notificationId}")
    public ResponseEntity<String> markNotificationAsReadByID(@PathVariable("notificationId") Long notificationId) {
        return new ResponseEntity<>(notificationService.markNotificationAsReadByID(notificationId), org.springframework.http.HttpStatus.OK);
    }

    @DeleteMapping("/delete/{notificationId}")
    public ResponseEntity<String> deleteNotification(@PathVariable("notificationId") Long notificationId) {
        return new ResponseEntity<>(notificationService.deleteNotificationByID(notificationId), org.springframework.http.HttpStatus.OK);
    }
}
