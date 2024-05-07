package com.bio.priovar.controllers;

import com.bio.priovar.models.InformationRequest;
import com.bio.priovar.services.InformationRequestService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/request")
@CrossOrigin
public class InformationRequestController{

    private final InformationRequestService informationRequestService;

    @Autowired
    public InformationRequestController(InformationRequestService informationRequestService){
        this.informationRequestService = informationRequestService;
    }

    @PostMapping("/send")
    public ResponseEntity<String> sendInformationRequest(@RequestParam Long clinicianId, @RequestParam Long patientId,@RequestParam String requestDescription) {
         return informationRequestService.sendInformationRequest(clinicianId, patientId, requestDescription);
    }

    @PostMapping("/accept/{informationRequestId}")
    public ResponseEntity<String> acceptInformationRequest(@PathVariable Long informationRequestId, @RequestParam String notificationAppendix) {
         return informationRequestService.acceptInformationRequest(informationRequestId, notificationAppendix);
     }

     @PostMapping("/reject/{informationRequestId}")
     public ResponseEntity<String> rejectInformationRequest(@PathVariable Long informationRequestId, @RequestParam String notificationAppendix) {
         return informationRequestService.rejectInformationRequest(informationRequestId, notificationAppendix);
     }

    @GetMapping("/waiting/{clinicianId}")
    public List<InformationRequest> getWaitingInformationRequests(@PathVariable Long clinicianId) {
         return informationRequestService.getWaitingInformationRequests(clinicianId);
    }

    @GetMapping("/pending/{medicalCenterId})")
    public List<InformationRequest> getPendingInformationRequests(@PathVariable Long medicalCenterId) {
         return informationRequestService.getPendingInformationRequests(medicalCenterId);
    }




}
