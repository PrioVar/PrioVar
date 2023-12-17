package com.bio.priovar.models;

public class BasicHPOCosineSimilarity implements  SimilarityStrategy{


    @Override
    public float calculateSimilarity(Patient primaryPatient, Patient secondaryPatient) {

        //check if patients have phenoqtype vector
        if(primaryPatient.getPhenotypeVector() == null || secondaryPatient.getPhenotypeVector() == null){
            // throw error
            throw new NullPointerException("Patient phenotype vector is null. Cannot calculate similarity.");
        }

        //calculate coisine similarity
        float dotProduct = 0;

        float[] vector1 = primaryPatient.getPhenotypeVector();
        float[] vector2 = secondaryPatient.getPhenotypeVector();

        for(int i = 0; i < primaryPatient.getPhenotypeVector().length; i++){
            dotProduct += vector1[i] * vector2[i];
        }

        float magnitude1 = 0;
        float magnitude2 = 0;

        for(int i = 0; i < vector1.length; i++){
            magnitude1 += Math.pow(vector1[i], 2);
            magnitude2 += Math.pow(vector2[i], 2);
        }

        magnitude1 = (float) Math.sqrt(magnitude1);
        magnitude2 = (float) Math.sqrt(magnitude2);

        float cosineSimilarity = dotProduct / (magnitude1 * magnitude2);

        return cosineSimilarity;
    }
}
