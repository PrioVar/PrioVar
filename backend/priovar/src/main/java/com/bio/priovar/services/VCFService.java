package com.bio.priovar.services;

import com.bio.priovar.models.*;
import com.bio.priovar.repositories.PatientRepository;
import com.bio.priovar.repositories.VCFRepository;
import org.apache.coyote.Response;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;


@Service
public class VCFService {
    private final VCFRepository vcfRepository;
    private final ClinicianService clinicianService;
    private final PatientRepository patientRepository;


    @Autowired
    public VCFService(VCFRepository vcfRepository, ClinicianService clinicianService, PatientRepository patientRepository) {

        this.patientRepository = patientRepository;
        this.clinicianService = clinicianService;
        this.vcfRepository = vcfRepository;
    }

    public ResponseEntity<String> uploadVCF(String base64Data, Long clinicianId, String patientName, int patientAge, String patientGender) {
        //First create a new patient
        Patient patient = new Patient();
        patient.setAge(patientAge);
        patient.setSex(patientGender);
        patient.setName(patientName);

        VCFFile vcfFile = new VCFFile();
        vcfFile.setContent(base64Data);
        String fileName = patientName + "_" + patientAge + "_" + patientGender + ".vcf";
        vcfFile.setFileName(fileName);

        List<ClinicianComment> clinicianComments = new ArrayList<>();
        vcfFile.setClinicanComments(clinicianComments);

        vcfRepository.save(vcfFile);
        patient.setVcfFile(vcfFile);

        //get clinician by clinicianId and add the patient into the clinician's patient list
        clinicianService.addPatientToClinician(clinicianId, patient);

        //Get the medical center of the clinician and set it to the patient
        MedicalCenter medicalCenter = clinicianService.getClinicianMedicalCenterByClinicianId(clinicianId);
        patient.setMedicalCenter(medicalCenter);
        //Save the patient
        patientRepository.save(patient);

        return ResponseEntity.ok("VCF File uploaded successfully");


    }
}
