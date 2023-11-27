package com.bio.priovar.services;

import com.bio.priovar.models.PhenotypeTerm;
import com.bio.priovar.repositories.PhenotypeTermRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class PhenotypeTermService {
    private final PhenotypeTermRepository phenotypeTermRepository;

    @Autowired
    public PhenotypeTermService(PhenotypeTermRepository phenotypeTermRepository) {
        this.phenotypeTermRepository = phenotypeTermRepository;
    }

    public String addPhenotypeTerm(PhenotypeTerm phenotypeTerm) {
        Optional<PhenotypeTerm> phenotypeTermOptional = phenotypeTermRepository.findPhenotypeTermById((long) phenotypeTerm.getId());

        if ( phenotypeTermOptional.isPresent() ) {
            return "HPO Term with ID: " + phenotypeTerm.getId() + " already exists";
        }

        phenotypeTermRepository.save(phenotypeTerm);
        return "Phenotype term added successfully";
    }

    public List<PhenotypeTerm> getAllPhenotypeTerms() {
        return phenotypeTermRepository.findAll();
    }

    public PhenotypeTerm getPhenotypeTermById(Long id) {
        return phenotypeTermRepository.findById(id).orElse(null);
    }

    /**public PhenotypeTerm getPhenotypeTermByHpoId(String hpoId) {
        return phenotypeTermRepository.findPhenotypeTermById(Long.valueOf(hpoId)).orElse(null);
    }*/
}
