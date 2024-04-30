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
@Node("GraphChat")
public class GraphChat {
    @Id
    @GeneratedValue
    private Long id;
    private String question;
    private String timestamp;
    private String response;

    @Relationship(type = "HAS_GRAPH_CHAT", direction = Relationship.Direction.OUTGOING)
    private MedicalCenter medicalCenter;
}
