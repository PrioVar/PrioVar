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
@Node("Chat")
public class Chat {
    @Id
    @GeneratedValue
    private Long id;
    private String question;
    private String timestamp;
    private String pico_clinical_question;
    private int article_count;
    private List<String> article_titles;
    private String RAG_GPT_output;

    @Relationship(type = "HAS_CHAT", direction = Relationship.Direction.OUTGOING)
    private MedicalCenter medicalCenter;
}
