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
@Node("Gene")
public class Gene {
    @Id
    @GeneratedValue
    private Long id;
    private String geneSymbol;

    @Relationship(type = "GENE_ASSOCIATED_WITH_PHENOTYPE", direction = Relationship.Direction.OUTGOING)
    private List<PhenotypeTerm> phenotypeTerms;

    // variant relationship: HAS_VARIANT_ON
    @Relationship(type = "HAS_VARIANT_ON", direction = Relationship.Direction.OUTGOING)
    private List<Variant> variants;

}
