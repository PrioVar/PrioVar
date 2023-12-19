package com.bio.priovar.controllers;
import com.bio.priovar.models.Patient;
import com.bio.priovar.services.QueryService;
import org.springframework.beans.factory.annotation.Autowired;

import com.bio.priovar.models.Query;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@CrossOrigin
@RequestMapping("/customQuery")
public class QueryController {

    private final QueryService queryService;

    @Autowired
    public QueryController(QueryService queryService) {
        this.queryService = queryService;
    }
    @PostMapping
    public List<Patient> runCustomQuery(@RequestBody Query query) {
        return queryService.runCustomQuery(query);
    }

}
