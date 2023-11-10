package com.bio.priovar.models;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
@Table
@Entity
public class Variant {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long ID;

    @OneToOne
    private Patient patient;
    private int CHROM;
    private int POS;
    private String ID_;
    private String REF;
    private String ALT;
    private String QUAL;
    private String FILTER;
    private String INFO;
}
