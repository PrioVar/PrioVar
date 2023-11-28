package com.bio.priovar.controllers;

import com.bio.priovar.models.PhenotypeTerm;
import com.bio.priovar.services.PhenotypeTermService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/phenotypeTerm")
@CrossOrigin
public class PhenotypeTermController {
    private final PhenotypeTermService phenotypeTermService;

    @Autowired
    public PhenotypeTermController(PhenotypeTermService phenotypeTermService) {
        this.phenotypeTermService = phenotypeTermService;
    }

    @GetMapping("/{phenotypeTermId}")
    public PhenotypeTerm getPhenotypeTermById(@PathVariable("phenotypeTermId") Long id) {
        return phenotypeTermService.getPhenotypeTermById(id);
    }

    /**@GetMapping("/byHpoId/{hpoId}")
    public PhenotypeTerm getPhenotypeTermByHpoId(@PathVariable("hpoId") String hpoId) {
        return phenotypeTermService.getPhenotypeTermByHpoId(hpoId);
    }*/

    @GetMapping()
    public List<PhenotypeTerm> getAllPhenotypeTerms() {
        return phenotypeTermService.getAllPhenotypeTerms();
    }


    @PostMapping("/add")
    public ResponseEntity<String> addPhenotypeTerm(@RequestBody PhenotypeTerm phenotypeTerm) {
        return new ResponseEntity<>(phenotypeTermService.addPhenotypeTerm(phenotypeTerm), org.springframework.http.HttpStatus.OK);
    }
}
