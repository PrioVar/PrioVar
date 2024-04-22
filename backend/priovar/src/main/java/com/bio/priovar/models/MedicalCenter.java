package com.bio.priovar.models;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.data.neo4j.core.schema.*;

@Getter
@Setter
@NoArgsConstructor
@Node("MedicalCenter")
public class MedicalCenter {

    @Id
    @GeneratedValue
    private Long id;

    private String name;
    private String address;
    private String phone;
    private String email;
    private String password;
    private Subscription subscription;
    private int remainingAnalyses;
}
