package com.bio.priovar.models;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.data.neo4j.core.schema.GeneratedValue;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.Node;
import org.springframework.data.neo4j.core.schema.Relationship;

import java.time.LocalDate;

@Getter
@Setter
@NoArgsConstructor
@Node("Notification")
public class Notification {
    @Id
    @GeneratedValue
    private Long id;

    private Boolean isRead;
    private String content;
    private LocalDate date;

    @Relationship(type="MEDICAL_CENTER_NOTIFICATION")
    private MedicalCenter medicalCenter;
}
