package com.bio.priovar.services;

import com.bio.priovar.models.*;
import com.bio.priovar.models.VCFFile.FileStatus;
import com.bio.priovar.models.dto.VCFFileDTO;
import com.bio.priovar.repositories.ClinicianRepository;
import com.bio.priovar.repositories.PatientRepository;
import com.bio.priovar.repositories.VCFRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.UUID;


@Service
public class VCFService {
    private final VCFRepository vcfRepository;
    private final ClinicianService clinicianService;
    private final MedicalCenterService medicalCenterService;
    private final ClinicianRepository clinicianRepository;
    private final PatientRepository patientRepository;

    @Autowired
    public VCFService(VCFRepository vcfRepository, ClinicianService clinicianService, 
                    MedicalCenterService medicalCenterService, ClinicianRepository clinicianRepository, 
                    PatientRepository patientRepository) {
        this.clinicianService = clinicianService;
        this.vcfRepository = vcfRepository;
        this.medicalCenterService = medicalCenterService;
        this.clinicianRepository = clinicianRepository;
        this.patientRepository = patientRepository;
    }

    public ResponseEntity<Long> uploadVCF(String base64Data, Long clinicianId, Long medicalCenterId) {

        VCFFile vcfFile = new VCFFile();
        vcfFile.setContent(base64Data);
        vcfFile.setFileStatus(FileStatus.FILE_WAITING);

        // generate an uuid file name
        String fileName = UUID.randomUUID().toString();
        vcfFile.setFileName(fileName);

        vcfFile.setMedicalCenter(medicalCenterService.getMedicalCenterById(medicalCenterId));

        List<ClinicianComment> clinicianComments = new ArrayList<>();
        vcfFile.setClinicianComments(clinicianComments);
        vcfFile.setCreatedAt(OffsetDateTime.now(ZoneOffset.ofHours(3)));
        vcfFile.setFinishedAt(null);

        // add it to files of clinician
        Clinician clinician = clinicianService.getClinicianById(clinicianId);
        if (clinician == null) {
            return ResponseEntity.badRequest().body(-1L);
        }
        clinician.getVcfFiles().add(vcfFile);
        clinicianRepository.save(clinician);
        
        // Save the vcf file and get the saved entity with ID populated
        VCFFile savedVcfFile = vcfRepository.save(vcfFile);

        // Check if the saved entity is not null and has an ID
        if (savedVcfFile != null && savedVcfFile.getId() != null) {
            // Return the ID of the newly created file in the response
            return ResponseEntity.ok(savedVcfFile.getId());
        } else {
            // Handle the error case where the file wasn't saved correctly
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    public List<VCFFileDTO> getVCFFilesByMedicalCenterId(Long medicalCenterId) {

        List<VCFFile> vcfFiles =  vcfRepository.findAllByMedicalCenterId(medicalCenterId);
        List<VCFFileDTO> vcfFileDTOs = new ArrayList<>();
        for (VCFFile vcfFile : vcfFiles) {
            Clinician clinician = clinicianRepository.findByVcfFilesId(vcfFile.getId()).orElse(null);
            String clinicianName = (clinician != null) ? clinician.getName() : "";

            VCFFileDTO vcfFileDTO = new VCFFileDTO(vcfFile.getId(), 
                                            vcfFile.getFileName(), vcfFile.getClinicianComments(), 
                                            clinicianName,
                                            vcfFile.getFileStatus(),
                                            vcfFile.getCreatedAt(),
                                            vcfFile.getFinishedAt());
            vcfFileDTOs.add(vcfFileDTO);
        }
        return vcfFileDTOs;
    }

    public List<VCFFileDTO> getVCFFilesByClinicianId(Long clinicianId) {
        Clinician clinician = clinicianRepository.findById(clinicianId).orElse(null);
        List<VCFFile> vcfFiles = clinician.getVcfFiles();
        List<VCFFileDTO> vcfFileDTOs = new ArrayList<>();

        for (VCFFile vcfFile : vcfFiles) {
            VCFFileDTO vcfFileDTO = new VCFFileDTO(vcfFile.getId(), 
                                        vcfFile.getFileName(), 
                                        vcfFile.getClinicianComments(), 
                                        clinician.getName(),
                                        vcfFile.getFileStatus(),
                                        vcfFile.getCreatedAt(),
                                        vcfFile.getFinishedAt());
            vcfFileDTOs.add(vcfFileDTO);
        }
        return vcfFileDTOs;
    }

    public ResponseEntity<String> addNoteToVCF(Long vcfFileId, Long clinicianId, String notes) {
        Optional<VCFFile> vcfFileOptional = vcfRepository.findById(vcfFileId);
        if (vcfFileOptional.isEmpty()) {
            return ResponseEntity.badRequest().body("VCF File with id " + vcfFileId + " does not exist");
        }
        
        VCFFile vcfFile = vcfFileOptional.get();
        ClinicianComment clinicianComment = new ClinicianComment();
        clinicianComment.setComment(notes);
        clinicianComment.setClinician(clinicianService.getClinicianById(clinicianId));
        
        // Add clinician comment to the VCF file
        vcfFile.addClinicianComment(clinicianComment);
        
        // Save the updated VCF file
        vcfRepository.save(vcfFile);
        
        return ResponseEntity.ok("Note added successfully");
    }

    public ResponseEntity<String> deleteVCFFile(Long vcfFileId) {
        Optional<VCFFile> vcfFileOptional = vcfRepository.findById(vcfFileId);
        if (vcfFileOptional.isEmpty()) {
            return ResponseEntity.badRequest().body("VCF File with id " + vcfFileId + " does not exist");
        }
        
        VCFFile vcfFile = vcfFileOptional.get();
        Patient patient = patientRepository.findByVcfFileId(vcfFileId);
        // Delete the VCF file
        vcfRepository.delete(vcfFile);
        patientRepository.delete(patient);
        return ResponseEntity.ok("VCF file deleted successfully");
    }
    

}
