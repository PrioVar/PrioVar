package com.bio.priovar.services;

import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
public class GraphLoaderService {

    public String startHPODataLoading() {
        RestTemplate restTemplate = new RestTemplate();
        String flaskUrl = "http://localhost:5001/load-hpo";
        String response = restTemplate.getForObject(flaskUrl, String.class);
        return response;
    }

    public String startDiseaseDataLoading() {
        RestTemplate restTemplate = new RestTemplate();
        String flaskUrl = "http://localhost:5001/load-diseases";
        String response = restTemplate.getForObject(flaskUrl, String.class);
        return response;
    }
}

