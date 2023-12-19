package com.bio.priovar.services;

import com.bio.priovar.models.Gene;
import com.bio.priovar.repositories.GeneRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class GeneService {

    private final GeneRepository geneRepository;

    @Autowired
    public GeneService(GeneRepository geneRepository) {
        this.geneRepository = geneRepository;
    }

    public List<Gene> getAllGenes() {
        return geneRepository.findAll();
    }
}

