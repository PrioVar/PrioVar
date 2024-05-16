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

    //*Columns:  Index(['#Uploaded_variation', 'Allele', 'Gene', 'Feature', 'Consequence',
    //       'Existing_variation', 'SYMBOL', 'CANONICAL', 'SIFT', 'PolyPhen',
    //       'HGVSp', 'AF', 'gnomADe_AF', 'CLIN_SIG', 'REVEL', 'SpliceAI_pred',
    //       'DANN_score', 'MetaLR_score', 'CADD_raw_rankscore', 'ExAC_AF',
    //       'ALFA_Total_AF', 'AlphaMissense_score', 'AlphaMissense_pred',
    //       'turkishvariome_TV_AF', 'HGSVc_original', 'HGSVp_original',
    //       'HGVSc_number', 'HGVSc_change', 'HGVSp_number', 'HGVSp_change',
    //       'SpliceAI_pred_symbol', 'DS_AG', 'DS_AL', 'DS_DG', 'DS_DL', 'DP_AG',
    //       'DP_AL', 'DP_DG', 'DP_DL', 'AlphaMissense_score_mean',
    //       'AlphaMissense_std_dev', 'AlphaMissense_pred_A', 'AlphaMissense_pred_B',
    //       'AlphaMissense_pred_P', 'PolyPhen_number', 'SIFT_number',
    //       'scaled_average_dot_product', 'scaled_min_dot_product',
    //       'scaled_max_dot_product', 'average_dot_product', 'min_dot_product',
    //       'max_dot_product', 'Priovar_score'],
    //
    //
    // add gene, HGSVc_original, HGVSp_original, Consequence, SYMBOL, turkishvariome_TV_AF_original, CLIN_SIG,  Priovar_score, AlphaMissense_score, AlphaMissense_pred, PolyPhen, SIFT, REVEL, SpliceAI_pred, DANN_score, MetaLR_score, CADD_raw_rankscore, ExAC_AF, ALFA_Total_AF, SpliceAI_pred_symbol, DS_AG, DS_AL, DS_DG, DS_DL, DP_AG, DP_AL, DP_DG, DP_DL, AlphaMissense_score_mean
    private String allele;
    private String consequence;

    private String symbol;

    private String gene;

    private String hgsvc_original;

    private String hgsvp_original;

    private String clin_sig;

    private String turkishvariome_tv_af_original;

    private Double priovar_score;

    private String alpha_missense_score_mean;

    //private String string_alpha_missense_score_mean;



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
