package com.bio.priovar.models.dto;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class PatientDTO {
    
    private Long patientId;
    private String patientName;
    private String clinicianName;
    private VCFFileDTO file;

    public PatientDTO(Long patientId, 
                    String patientName, 
                    String clinicianName,
                    VCFFileDTO file) {
        this.patientId = patientId;
        this.patientName = patientName;
        this.clinicianName = clinicianName;
    }
}
