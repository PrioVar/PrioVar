package com.bio.priovar.models;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class Query {

    private String sex;
    private Gene[] genes;
    private PhenotypeTerm[] phenotypeTerms;
    private int ageIntervalStart;
    private int ageIntervalEnd;



}
