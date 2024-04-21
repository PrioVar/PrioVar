package com.bio.priovar.services;

import com.bio.priovar.repositories.ClinicianCommentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class ClinicianCommentService {
    private final ClinicianCommentRepository clinicianCommentRepository;
    @Autowired
    public ClinicianCommentService(ClinicianCommentRepository clinicianCommentRepository) {
        this.clinicianCommentRepository = clinicianCommentRepository;
    }

}
