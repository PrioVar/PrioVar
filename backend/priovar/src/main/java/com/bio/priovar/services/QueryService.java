package com.bio.priovar.services;

import com.bio.priovar.models.Patient;
import com.bio.priovar.models.Query;
import com.bio.priovar.repositories.PatientRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
public class QueryService {

    private final PatientRepository patientRepository;

    @Autowired
    public QueryService(PatientRepository patientRepository) {
        this.patientRepository = patientRepository;
    }

    public List<Patient> runCustomQuery(Query query) {

        String sexQuery = query.getSex();
        int ageIntervalStartQuery = query.getAgeIntervalStart();
        int ageIntervalEndQuery = query.getAgeIntervalEnd();

        //Get the query's genes
        List<String> genesQuery = new ArrayList<>();
        for (int i = 0; i < query.getGenes().length; i++) {
            genesQuery.add(query.getGenes()[i].getGeneSymbol());

        }

        //Get the query's phenotype terms
        List<String> phenotypeTermsQuery = new ArrayList<>();
        for (int i = 0; i < query.getPhenotypeTerms().length; i++) {
            phenotypeTermsQuery.add(query.getPhenotypeTerms()[i].getName());
        }

        List<Patient> patients = patientRepository.findPatientsBySexAndAgeIntervalAndGenesAndPhenotypeTerms(sexQuery, ageIntervalStartQuery, ageIntervalEndQuery, genesQuery, phenotypeTermsQuery);

        return patients;

    }


}
