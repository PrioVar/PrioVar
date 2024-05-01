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
@Node("Clinician")
public class Clinician extends Actor{

    @Relationship(type="HAS_PATIENT")
    private List<Patient> patients;
    @Relationship(type="REQUESTED_FOR")
    private List<Patient> requestedPatients; //Patients requested and obtained by the clinician
    @Relationship(type="WORKS_AT")
    private MedicalCenter medicalCenter;

    @Relationship(type = "UPLOADED_FILE", direction = Relationship.Direction.OUTGOING)
    private List<VCFFile> vcfFiles;

}
