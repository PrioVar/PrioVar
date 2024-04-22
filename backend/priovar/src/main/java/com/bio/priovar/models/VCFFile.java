package com.bio.priovar.models;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.data.neo4j.core.schema.GeneratedValue;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.Node;
import org.springframework.data.neo4j.core.schema.Relationship;
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

    @Relationship(type = "HAS_COMMENTS", direction = Relationship.Direction.OUTGOING)
    private List<ClinicianComment> clinicianComments;

    @Relationship(type = "UPLOADED_BY", direction = Relationship.Direction.OUTGOING)
    private Clinician clinician;
}
