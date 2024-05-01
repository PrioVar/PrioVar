package com.bio.priovar.models;

import com.bio.priovar.serializers.PatientLiteSerializer;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.data.neo4j.core.schema.GeneratedValue;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.Relationship;

@Getter
@Setter
@NoArgsConstructor
public class PairSimilarity {

    @GeneratedValue
    @Id
    private Long id;

    @Relationship(type = "SIMILARITY_PRIMARY_PATIENT", direction = Relationship.Direction.OUTGOING)
    @JsonSerialize(using = PatientLiteSerializer.class)
    private Patient primaryPatient;

    @Relationship(type = "SIMILARITY_SECONDARY_PATIENT", direction = Relationship.Direction.OUTGOING)
    @JsonSerialize(using = PatientLiteSerializer.class)
    private Patient secondaryPatient;

    public enum REPORT_STATUS {
        PENDING,
        APPROVED,
    }

    private float genotypeScore;
    private float phenotypeScore;
    private float totalScore;
    private REPORT_STATUS status;
    private String similarityStrategy;
}
