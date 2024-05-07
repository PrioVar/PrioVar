package com.bio.priovar.models;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.data.neo4j.core.schema.GeneratedValue;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.Node;
import org.springframework.data.neo4j.core.schema.Relationship;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.List;


@Getter
@Setter
@NoArgsConstructor
@Node("VCFFile")
public class VCFFile {

    @GeneratedValue
    @Id
    private Long id;
    private String content;
    private String fileName;

    //api_file_id
    private String apiFileId;

    @Relationship(type = "HAS_COMMENTS", direction = Relationship.Direction.OUTGOING)
    private List<ClinicianComment> clinicianComments;

    @Relationship(type="FILE_BELONGS_TO_MEDICAL_CENTER")
    private MedicalCenter medicalCenter;

    // Enum to represent file status
    public enum FileStatus {
        ANALYSIS_DONE,
        ANALYSIS_IN_PROGRESS,
        FILE_ANNOTATED,
        FILE_WAITING
    }

    private FileStatus fileStatus;

    OffsetDateTime createdAt;
    OffsetDateTime finishedAt;

    public void addClinicianComment(ClinicianComment clinicianComment) {
        if(clinicianComments == null) {
            clinicianComments = new ArrayList<>();
        }
        clinicianComments.add(clinicianComment);
    }
}
