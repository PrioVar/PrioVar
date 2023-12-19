package com.bio.priovar.controllers;

import com.bio.priovar.services.GraphLoaderService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class GraphLoaderController {

    private final GraphLoaderService dataLoaderService;

    @Autowired
    public GraphLoaderController(GraphLoaderService dataLoaderService) {
        this.dataLoaderService = dataLoaderService;
    }

    @GetMapping("/load-data")
    public void loadData() {
        dataLoaderService.startHPODataLoading();
        System.out.println("HPO data loaded");
        dataLoaderService.startDiseaseDataLoading();
        System.out.println("Disease data loaded");
        dataLoaderService.startGeneDatafromHPOLoading();
        System.out.println("Gene data loaded");
    }
}

