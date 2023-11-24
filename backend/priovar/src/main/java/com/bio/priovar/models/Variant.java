package com.bio.priovar.models;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.data.neo4j.core.schema.GeneratedValue;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.Node;

@Getter
@Setter
@NoArgsConstructor
@Node("Variant")
public class Variant {

    @Id
    @GeneratedValue
    private Long id;

    private String chrom;
    private String pos;
    private String id_;
    private String ref;
    private String alt;
    private String qual;
    private String filter;
    private String info;

    private Boolean isClinVar;

    // create a sample Variant object in comments as a JSON object
    // {
    //     "chrom": "1",
    //     "pos": "100",
    //     "id_": "rs123",
    //     "ref": "A",
    //     "alt": "T",
    //     "qual": "100",
    //     "filter": "PASS",
    //     "info": "info",
    //     "isClinVar": true
    // }
}
