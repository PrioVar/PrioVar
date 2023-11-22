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
@Node("Patient")
public class Patient {

    @Id
    @GeneratedValue
    private Long id;

    @Relationship(type = "BELONGS_TO_MEDICAL_CENTER", direction = Relationship.Direction.OUTGOING)
    private MedicalCenter medicalCenter;

    @Relationship(type = "HAS_DISEASE", direction = Relationship.Direction.OUTGOING)
    private Disease disease;
}
