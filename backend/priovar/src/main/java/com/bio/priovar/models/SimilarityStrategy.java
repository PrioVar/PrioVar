package com.bio.priovar.models;

//strategy design pattern for similarity

public interface SimilarityStrategy {
    public float calculateSimilarity(Patient primaryPatient, Patient secondaryPatient);
}