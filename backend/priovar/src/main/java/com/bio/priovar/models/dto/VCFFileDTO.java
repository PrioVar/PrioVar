package com.bio.priovar.models.dto;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.List;

import com.bio.priovar.models.ClinicianComment;
import com.bio.priovar.models.VCFFile.FileStatus;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class VCFFileDTO {

    private Long vcfFileId;
    private String fileName;
    private List<String> clinicianComments;
    private String clinicianName;
    // Enum to represent file status
    
    private FileStatus fileStatus;
    private OffsetDateTime createdAt;
    private OffsetDateTime finishedAt;
    public VCFFileDTO(
            Long vcfFileId,
            String fileName,
            List<ClinicianComment> clinicianComments,
            String clinicianName,
            FileStatus fileStatus,
            OffsetDateTime createdAt,
            OffsetDateTime finishedAt
    ) {
        this.vcfFileId = vcfFileId;
        this.fileName = fileName;
        if(clinicianComments != null) {
            this.clinicianComments = new ArrayList<>();
            for (ClinicianComment comment : clinicianComments) {
                this.clinicianComments.add(comment.getComment());
            }
        }
        else {
            this.clinicianComments = null;
        }
        this.clinicianName = clinicianName;
        this.fileStatus = fileStatus;
        this.createdAt = createdAt;
        this.finishedAt = finishedAt;
    }
}
