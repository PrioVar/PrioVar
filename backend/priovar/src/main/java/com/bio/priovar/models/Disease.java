package com.bio.priovar.models;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.data.neo4j.core.schema.GeneratedValue;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.Node;
import org.springframework.data.neo4j.core.schema.Relationship;
import org.springframework.data.neo4j.core.support.UUIDStringGenerator;

import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@Node("Disease")
public class Disease {
    @Id
    @GeneratedValue
    private Long id;

    private String disease_name;
    private String database_id;

    @Relationship(type = "ASSOCIATED_WITH_PHENOTYPE", direction = Relationship.Direction.OUTGOING)
    private List<PhenotypeTerm> phenotypeTerms;



}
