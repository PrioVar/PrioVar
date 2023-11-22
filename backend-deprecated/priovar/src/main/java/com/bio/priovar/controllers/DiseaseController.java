package com.bio.priovar.controllers;

import com.bio.priovar.models.Disease;
import com.bio.priovar.services.DiseaseService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/disease")
@CrossOrigin
public class DiseaseController {

    private final DiseaseService diseaseService;

    @Autowired
    public DiseaseController(DiseaseService diseaseService) {
        this.diseaseService = diseaseService;
    }

    @GetMapping()
    public List<Disease> getAllDiseases() {
        return diseaseService.getAllDiseases();
    }

    @GetMapping("/{diseaseId}")
    public Disease getDiseaseById(@PathVariable("diseaseId") Long id) {
        return diseaseService.getDiseaseById(id);
    }

    @PostMapping("/add")
    public void addDisease(@RequestBody Disease Disease) {
        diseaseService.addDisease(Disease);
    }
}
