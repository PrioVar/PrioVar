package com.bio.priovar.services;

import com.bio.priovar.models.*;
import com.bio.priovar.repositories.InformationRequestRepository;
import com.bio.priovar.repositories.MedicalCenterRepository;
import com.bio.priovar.repositories.NotificationRepository;
import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;

import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.List;

@Service
public class InformationRequestService {

    private final InformationRequestRepository informationRequestRepository;
    private final NotificationRepository notificationRepository;
    private final ClinicianService clinicianService;
    private final PatientService patientService;
    private final MedicalCenterRepository medicalCenterRepository;
    @Autowired
    public InformationRequestService(InformationRequestRepository informationRequestRepository, NotificationRepository notificationRepository,
                                      ClinicianService clinicianService, PatientService patientService, MedicalCenterRepository medicalCenterRepository) {
        this.informationRequestRepository = informationRequestRepository;
        this.notificationRepository = notificationRepository;
        this.clinicianService = clinicianService;
        this.patientService = patientService;
        this.medicalCenterRepository = medicalCenterRepository;
    }

    public ResponseEntity<String> sendInformationRequest(Long clinicianId, Long patientId, String requestDescription) {
        InformationRequest informationRequest = new InformationRequest();
        //Check if the clinician and patient exist
        Clinician clinician = clinicianService.getClinicianById(clinicianId);
        if (clinician == null) {
            return ResponseEntity.badRequest().body("Clinician not found");
        }
        Patient patient = patientService.getPatientById(patientId);
        if (patient == null) {
            return ResponseEntity.badRequest().body("Patient not found");
        }

        informationRequest.setClinician(clinician);
        informationRequest.setPatient(patient);
        informationRequest.setIsPending(true);
        informationRequest.setIsApproved(false);
        informationRequest.setIsRejected(false);
        informationRequest.setRequestDescription(requestDescription);

        MedicalCenter receivingMedicalCenter = patient.getMedicalCenter();
        informationRequestRepository.save(informationRequest);


        //Create a notification object for the medical center
        Notification notification = new Notification();
        notification.setType("REQUEST");
        notification.setSender(informationRequest.getClinician());
        notification.setReceiver(receivingMedicalCenter);
        notification.setSendAt(OffsetDateTime.now(ZoneOffset.ofHours(3)));
        notification.setNotification("A new information request has been made for patient " + patient.getName() +
                                     " by clinician " + informationRequest.getClinician().getName() +
                                     " from " + informationRequest.getClinician().getMedicalCenter().getName() +
                                     " Health Center!");
        notification.setInformationRequest(informationRequest);
       
        notification.setAppendix(informationRequest.getRequestDescription());
        notification.setIsRead(false);

        notificationRepository.save(notification);
        return ResponseEntity.ok("Information request sent successfully");
    }

    public ResponseEntity<String> acceptInformationRequest(Long informationRequestId, String notificationAppendix) {
        InformationRequest informationRequest = informationRequestRepository.findById(informationRequestId).orElse(null);
        if (informationRequest == null) {
            return ResponseEntity.badRequest().body("Information request not found");
        }
        informationRequest.setIsApproved(true);
        informationRequest.setIsPending(false);
        informationRequest.setIsRejected(false);

        informationRequestRepository.save(informationRequest);

        //Create a notification object for the clinician
        Notification notification = new Notification();
        notification.setType("RESPONSE");
        notification.setSender(informationRequest.getPatient().getMedicalCenter());
        notification.setReceiver(informationRequest.getClinician());
        notification.setSendAt(OffsetDateTime.now(ZoneOffset.ofHours(3)));
        notification.setNotification("Your information request for patient " + informationRequest.getPatient().getName() +
                                     " has been accepted by " + informationRequest.getPatient().getMedicalCenter().getName() +
                                     " Health Center!");

        notification.setIsRead(false);
        notification.setAppendix(notificationAppendix);
        Clinician clinician = informationRequest.getClinician();

        notificationRepository.save(notification);
        return clinicianService.addRequestedPatientToClinician(clinician, informationRequest.getPatient());
    }

    public ResponseEntity<String> rejectInformationRequest(Long informationRequestId, String notificationAppendix) {
        InformationRequest informationRequest = informationRequestRepository.findById(informationRequestId).orElse(null);
        if (informationRequest == null) {
            return ResponseEntity.badRequest().body("Information request not found");
        }
        informationRequest.setIsApproved(false);
        informationRequest.setIsPending(false);
        informationRequest.setIsRejected(true);

        //Create a notification object for the clinician
        Notification notification = new Notification();
        notification.setType("RESPONSE");
        notification.setSender(informationRequest.getPatient().getMedicalCenter());
        notification.setReceiver(informationRequest.getClinician());
        notification.setSendAt(OffsetDateTime.now(ZoneOffset.ofHours(3)));
        notification.setNotification("Your information request for patient Gender: " + informationRequest.getPatient().getSex() + " Age: " + informationRequest.getPatient().getAge() +
                                     " has been rejected by " + informationRequest.getPatient().getMedicalCenter().getName() + " Health Center!");
        notification.setIsRead(false);
        notification.setAppendix(notificationAppendix);

        notificationRepository.save(notification);
        informationRequestRepository.save(informationRequest);
        return ResponseEntity.ok("Information request rejected successfully");
    }

    public List<InformationRequest> getPendingInformationRequests(Long medicalCenterId) {
        //Find medical center by id
        MedicalCenter medicalCenter = medicalCenterRepository.findById(medicalCenterId).orElse(null);
        if (medicalCenter == null) {
            return null;
        }
        List<InformationRequest> informationRequests = informationRequestRepository.findAllByPatientMedicalCenterAndIsPending(medicalCenter, true);
        return informationRequests;
    }

    public List<InformationRequest> getWaitingInformationRequests(Long clinicianId) {
        List<InformationRequest> informationRequests = informationRequestRepository.findAllByClinicianIdAndIsPending(clinicianId, true);
        return informationRequests;
    }


}
