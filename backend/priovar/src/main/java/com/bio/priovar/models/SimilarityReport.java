package com.bio.priovar.models;

//

import com.bio.priovar.serializers.PatientLiteSerializer;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.data.neo4j.core.schema.GeneratedValue;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.Node;
import org.springframework.data.neo4j.core.schema.Relationship;

import java.time.OffsetDateTime;
import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@Node("SimilarityReport")
public class SimilarityReport {

    @GeneratedValue
    @Id
    private Long id;

    private OffsetDateTime createdAt;

    @Relationship(type = "SEARCHED_PATIENT", direction = Relationship.Direction.OUTGOING)
    @JsonSerialize(using = PatientLiteSerializer.class)
    private Patient primaryPatient;

    @Relationship(type = "CONTAINS_PAIR_SIMILARITIES", direction = Relationship.Direction.OUTGOING)
    private List<PairSimilarity> pairSimilarities;
}
