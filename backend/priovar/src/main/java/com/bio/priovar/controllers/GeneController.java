package com.bio.priovar.controllers;

import com.bio.priovar.models.Gene;
import com.bio.priovar.services.GeneService;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/gene")
@CrossOrigin
public class GeneController {

    private final GeneService geneService;

    public GeneController(GeneService geneService) {
        this.geneService = geneService;
    }

    @GetMapping()
    public List<Gene> getAllGenes() {
        return geneService.getAllGenes();
    }

}
