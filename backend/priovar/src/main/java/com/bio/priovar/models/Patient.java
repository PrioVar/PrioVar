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
@Node("Patient")
public class Patient {

    @Id
    @GeneratedValue
    private Long id;

    private String name;
    private int age;
    private String sex;
    private String clinicalHistory;
    private String genesusId;

    // vectorized form of the patient's phenotype (float array)
    private float[] phenotypeVector;

    @Relationship(type = "BELONGS_TO_MEDICAL_CENTER", direction = Relationship.Direction.OUTGOING)
    private MedicalCenter medicalCenter;

    @Relationship(type = "HAS_DISEASE", direction = Relationship.Direction.OUTGOING)
    private Disease disease;

    @Relationship(type = "HAS_VARIANT", direction = Relationship.Direction.OUTGOING)
    private List<Variant> variants;

    @Relationship(type = "HAS_PHENOTYPE_TERM", direction = Relationship.Direction.OUTGOING)
    private List<PhenotypeTerm> phenotypeTerms;

}
