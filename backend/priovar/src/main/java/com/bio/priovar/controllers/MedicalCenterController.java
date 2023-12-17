package com.bio.priovar.controllers;

import com.bio.priovar.models.MedicalCenter;
import com.bio.priovar.models.dto.LoginObject;
import com.bio.priovar.services.MedicalCenterService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/medicalCenter")
@CrossOrigin
public class MedicalCenterController {
    private final MedicalCenterService medicalCenterService;

    @Autowired
    public MedicalCenterController(MedicalCenterService medicalCenterService) {
        this.medicalCenterService = medicalCenterService;
    }

    @GetMapping()
    public List<MedicalCenter> getAllMedicalCenters() {
        return medicalCenterService.getAllMedicalCenters();
    }

    @GetMapping("/{medicalCenterId}")
    public MedicalCenter getMedicalCenterById(@PathVariable("medicalCenterId") Long medicalCenterId) {
        return medicalCenterService.getMedicalCenterById(medicalCenterId);
    }

    @PostMapping("/login")
    public ResponseEntity<LoginObject> loginMedicalCenter(@RequestParam String email, @RequestParam String password ) {
        return medicalCenterService.loginMedicalCenter(email,password);
    }

    @PatchMapping("/changePassword")
    public ResponseEntity<String> changePasswordMedicalCenter(@RequestParam String email, @RequestParam String newPass, @RequestParam String oldPass) {
        return medicalCenterService.changePasswordByEmailMedicalCenter(email,newPass, oldPass);
    }

    @PostMapping("/add")
    public void addMedicalCenter(@RequestBody MedicalCenter medicalCenter) {
        medicalCenterService.addMedicalCenter(medicalCenter);
    }

    @PostMapping("/addSubscription/{medicalCenterId}/{subscriptionId}")
    public ResponseEntity<String> addSubscriptionToMedicalCenter(@PathVariable("medicalCenterId") Long medicalCenterId, @PathVariable("subscriptionId") Long subscriptionId) {
        return medicalCenterService.addSubscriptionToMedicalCenter(medicalCenterId, subscriptionId);
    }
}
