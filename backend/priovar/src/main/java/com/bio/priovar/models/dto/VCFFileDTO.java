package com.bio.priovar.models.dto;

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
        for (ClinicianComment comment : clinicianComments) {
            this.clinicianComments.add(comment.getComment());
        }
        this.clinicianName = clinicianName;
    }
}
