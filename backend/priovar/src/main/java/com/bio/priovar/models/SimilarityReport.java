package com.bio.priovar.models;

//

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.data.neo4j.core.schema.GeneratedValue;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.Node;

@Getter
@Setter
@NoArgsConstructor
@Node("SimilarityReport")
public class SimilarityReport {

    @GeneratedValue
    @Id
    private Long id;
    private Patient primaryPatient;
    private Patient secondaryPatient;
    private float genotypeScore;
    private float phenotypeScore;
    private float totalScore;

    private String status;

    //similarity strategy
    private String similarityStrategy;

}
