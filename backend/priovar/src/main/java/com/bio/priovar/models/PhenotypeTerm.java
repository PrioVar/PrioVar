package com.bio.priovar.models;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.data.neo4j.core.schema.GeneratedValue;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.Node;

import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@Node("PhenotypeTerm")
public class PhenotypeTerm {
    @Id
    @GeneratedValue
    private Long id;

    private String hpoId;
    private String name;
    private List<String> altIds;
    private String definition;
    private List<String> synonyms;
    private String comment;
    private List<String> xrefs;
    private List<String> isAs;

    // override equals method by comparing hpoId
    @Override
    public boolean equals(Object obj) {
        if (obj == this) return true;
        if (!(obj instanceof PhenotypeTerm)) return false;
        PhenotypeTerm phenotypeTerm = (PhenotypeTerm) obj;
        return phenotypeTerm.getHpoId().equals(this.hpoId);
    }

    // create a sample PhenotypeTerm object in comments as a JSON object
    // {
    //     "hpoId": "HP:0000001",
    //     "name": "All",
    //     "altIds": [
    //         "HP:0000005",
    //         "HP:0000118"
    //     ],
    //     "definition": "All",
    //     "synonyms": [
    //         "All",
    //         "All"
    //     ],
    //     "comment": "All",
    //     "xrefs": [
    //         "All",
    //         "All"
    //     ],
    //     "isAs": [
    //         "All",
    //         "All"
    //     ]
    // }
}
