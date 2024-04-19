package com.bio.priovar.services;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
public class GraphLoaderService {

    @Value("${flask.url}")
    private String flaskUrl;

    public String startHPODataLoading() {
        RestTemplate restTemplate = new RestTemplate();
        String loadHPOUrl = flaskUrl + "/load-hpo";
        String response = restTemplate.getForObject(loadHPOUrl, String.class);
        return response;
    }

    public String startDiseaseDataLoading() {
        RestTemplate restTemplate = new RestTemplate();
        String loadDiseasesUrl = flaskUrl + "/load-diseases";
        String response = restTemplate.getForObject(loadDiseasesUrl, String.class);
        return response;
    }

    public String startGeneDatafromHPOLoading() {
        RestTemplate restTemplate = new RestTemplate();
        String loadGenesUrl = flaskUrl + "/load-genes";
        String response = restTemplate.getForObject(loadGenesUrl, String.class);
        return response;
    }
}

