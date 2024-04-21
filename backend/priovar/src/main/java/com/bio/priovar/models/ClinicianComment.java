package com.bio.priovar.models;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.data.neo4j.core.schema.GeneratedValue;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.Node;
import org.springframework.data.neo4j.core.schema.Relationship;


@Getter
@Setter
@NoArgsConstructor
@Node("ClinicianComment")
public class ClinicianComment {

    @GeneratedValue
    @Id
    private Long id;
    private String comment;

    @Relationship(type="COMMENTED", direction = Relationship.Direction.INCOMING)
    private Clinician clinician;





}
