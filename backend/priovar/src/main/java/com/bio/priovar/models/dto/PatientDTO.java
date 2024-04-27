package com.bio.priovar.models.dto;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class PatientDTO {
    
    private Long patientId;
    private String patientName;
    private int age;
    private String sex;
    private VCFFileDTO file;
    private Long clinicianId;
    public PatientDTO(Long patientId, 
                    String patientName, 
                    int age,
                    String sex,
                    VCFFileDTO file,
                    Long clinicianId) {
        this.patientId = patientId;
        this.patientName = patientName;
        this.age = age;
        this.sex = sex;
        this.file = file;
        this.clinicianId = clinicianId;
    }
}
