package com.bio.priovar.models;

import com.bio.priovar.serializers.PatientLiteSerializer;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
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
@Node("InformationRequest")
public class InformationRequest {

    @Id
    @GeneratedValue
    private Long id;
    private String requestDescription;
    private Boolean isApproved;
    private Boolean isRejected;
    private Boolean isPending;

    @Relationship(type = "REQUESTED_BY", direction = Relationship.Direction.OUTGOING)
    private Clinician clinician;

    @Relationship(type = "REQUESTED_FOR", direction = Relationship.Direction.OUTGOING)
    @JsonSerialize(using = PatientLiteSerializer.class)
    private Patient patient;

}
