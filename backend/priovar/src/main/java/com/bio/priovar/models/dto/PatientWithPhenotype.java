package com.bio.priovar.models.dto;

import com.bio.priovar.models.Patient;
import com.bio.priovar.models.PhenotypeTerm;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
@NoArgsConstructor
public class PatientWithPhenotype {
    private Patient patient;
    private List<PhenotypeTerm> phenotypeTerms;
    private Long vcfFileId;
    private Long clinicianId;
}
