package com.bio.priovar.models;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.Node;
import org.springframework.data.neo4j.core.schema.Relationship;
import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@Node("PhenotypeTerm")
public class PhenotypeTerm {
    @Id
    private Long id;
    private String name;//label
    //private List<String> altIds;
    private String definition;
    private List<String> synonyms;
    private String comment;
    private List<String> xrefs;

    //phenotypes are a directed acyclic graph
     @Relationship(type = "IS_A", direction = Relationship.Direction.OUTGOING)
     private List<PhenotypeTerm> parents;

    // gene associations
    @Relationship(type = "ASSOCIATED_WITH_GENE", direction = Relationship.Direction.OUTGOING)
    private List<Gene> genes;

    // disease associations
    @Relationship(type = "ASSOCIATED_WITH_DISEASE", direction = Relationship.Direction.OUTGOING)
    private List<Disease> diseases;



    @Override
    public boolean equals(Object obj) {
        if (obj == this) return true;
        if (!(obj instanceof PhenotypeTerm)) return false;
        PhenotypeTerm phenotypeTerm = (PhenotypeTerm) obj;
        return phenotypeTerm.id == this.id;
    }

    //to do
    public String getLink() {
        return "https://hpo.jax.org/app/browse/term/";
        //http://purl.obolibrary.org/obo/HP_
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
