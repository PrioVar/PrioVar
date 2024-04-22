package com.bio.priovar.models.dto;

import java.util.ArrayList;
import java.util.List;

import com.bio.priovar.models.ClinicianComment;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class VCFFileDTO {

    private Long vcfFileId;
    private String fileName;
    private List<String> clinicianComments;
    private String clinicianName;

    public VCFFileDTO(Long vcfFileId, 
                    String fileName, 
                    List<ClinicianComment> clinicianComments, 
                    String clinicianName) {
        this.vcfFileId = vcfFileId;
        this.fileName = fileName;
        if(clinicianComments != null) {
            this.clinicianComments = new ArrayList<>();
            for (ClinicianComment comment : clinicianComments) {
                this.clinicianComments.add(comment.getComment());
            }
        }
        else {
            System.out.println("clinicianComments is null");
            System.out.println("clinicianComments is null");
            System.out.println("clinicianComments is null");
            this.clinicianComments = null;
        }
        this.clinicianName = clinicianName;
    }
}
