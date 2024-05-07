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
@Node("Disease")
public class Disease {
    @Id
    @GeneratedValue
    private Long id;

    private String diseaseName;
    private String databaseId;

    @Relationship(type = "DISEASE_ASSOCIATED_WITH_PHENOTYPE", direction = Relationship.Direction.OUTGOING)
    private List<PhenotypeTerm> phenotypeTerms;



}
