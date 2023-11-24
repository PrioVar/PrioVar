package com.bio.priovar.controllers;

import com.bio.priovar.models.PhenotypeTerm;
import com.bio.priovar.services.PhenotypeTermService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/phenotypeTerm")
@CrossOrigin
public class PhenotypeTermController {
    private final PhenotypeTermService phenotypeTermService;

    @Autowired
    public PhenotypeTermController(PhenotypeTermService phenotypeTermService) {
        this.phenotypeTermService = phenotypeTermService;
    }


    @PostMapping("/add")
    public ResponseEntity<String> addPhenotypeTerm(@RequestBody PhenotypeTerm phenotypeTerm) {
        return new ResponseEntity<>(phenotypeTermService.addPhenotypeTerm(phenotypeTerm), org.springframework.http.HttpStatus.OK);
    }
}
