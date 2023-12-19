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
        List<String> genesQuery = new ArrayList<>();
        List<Long> phenotypeTermsQuery = new ArrayList<>();
        //Get the query's genes if query has genes
        if (query.getGenes() != null) {
            for (int i = 0; i < query.getGenes().length; i++) {
                genesQuery.add(query.getGenes()[i].getGeneSymbol());

            }
        }

        //Get the query's phenotype terms if query has phenotype terms
        if (query.getPhenotypeTerms() != null) {
            for (int i = 0; i < query.getPhenotypeTerms().length; i++) {
                phenotypeTermsQuery.add(query.getPhenotypeTerms()[i].getId());
            }
        }

        List<Patient> patients = patientRepository.findPatientsBySexAndAgeIntervalAndGenesAndPhenotypeTerms(sexQuery, ageIntervalStartQuery, ageIntervalEndQuery, genesQuery, phenotypeTermsQuery);

        return patients;

    }


}
