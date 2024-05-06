package com.bio.priovar.models;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.data.neo4j.core.schema.*;

@Getter
@Setter
@NoArgsConstructor
@Node("MedicalCenter")
public class MedicalCenter extends Actor {


    private String address;
    private String phone;
    private Subscription subscription;
    private int remainingAnalyses;
}
