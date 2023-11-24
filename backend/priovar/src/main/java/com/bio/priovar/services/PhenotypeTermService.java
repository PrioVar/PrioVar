package com.bio.priovar.services;

import com.bio.priovar.models.PhenotypeTerm;
import com.bio.priovar.repositories.PhenotypeTermRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Optional;

@Service
public class PhenotypeTermService {
    private final PhenotypeTermRepository phenotypeTermRepository;

    @Autowired
    public PhenotypeTermService(PhenotypeTermRepository phenotypeTermRepository) {
        this.phenotypeTermRepository = phenotypeTermRepository;
    }

    public String addPhenotypeTerm(PhenotypeTerm phenotypeTerm) {
        Optional<PhenotypeTerm> phenotypeTermOptional = phenotypeTermRepository.findPhenotypeTermByHpoId(phenotypeTerm.getHpoId());

        if ( phenotypeTermOptional.isPresent() ) {
            return "HPO Term with ID: " + phenotypeTerm.getHpoId() + " already exists";
        }

        phenotypeTermRepository.save(phenotypeTerm);
        return "Phenotype term added successfully";
    }
}
