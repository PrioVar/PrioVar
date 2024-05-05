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

    @GetMapping("/listNotifications/{actorId}")
    public List<Notification> getNotificationsByActorId(@PathVariable("actorId") Long actorId) {
        List<Notification> notifications = notificationService.getNotificationsByActorId(actorId);
        return notifications;
    }

    @GetMapping("/listNotifications/{actorId}/unread")
    public List<Notification> getUnreadNotificationsByActorId(@PathVariable("actorId") Long actorId) {
        return notificationService.getUnreadNotificationsByActorId(actorId);
    }

    @PostMapping("/markRead/{notificationId}")
    public ResponseEntity<String> markNotificationAsReadByID(@PathVariable("notificationId") Long notificationId) {
        return  notificationService.markNotificationAsReadByID(notificationId);
    }

    @PostMapping("/markAllRead/{actorId}")
    public ResponseEntity<String> markAllNotificationsAsReadByActorId(@PathVariable("actorId") Long actorId) {
        return notificationService.markAllNotificationsAsReadByActorId(actorId);
    }

    @DeleteMapping("/delete/{notificationId}")
    public ResponseEntity<String> deleteNotification(@PathVariable("notificationId") Long notificationId) {
        return new ResponseEntity<>(notificationService.deleteNotificationByID(notificationId), org.springframework.http.HttpStatus.OK);
    }
}
