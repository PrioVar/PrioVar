package com.bio.priovar.services;

import com.bio.priovar.models.Admin;
import com.bio.priovar.models.dto.LoginObject;
import com.bio.priovar.repositories.AdminRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

@Service
public class AdminService {
    private final AdminRepository adminRepository;

    @Autowired
    public AdminService(AdminRepository adminRepository) {
        this.adminRepository = adminRepository;
    }

    public ResponseEntity<String> addAdmin(Admin admin) {
        Admin adminWithSameEmail = adminRepository.findAdminByEmail(admin.getEmail());

        if (adminWithSameEmail != null) {
            return ResponseEntity.badRequest().body("Admin with same email already exists!");
        }

        adminRepository.save(admin);
        return ResponseEntity.ok("Admin added successfully!");
    }

    public ResponseEntity<LoginObject> loginAdmin(String email, String password) {
        Admin admin = adminRepository.findAdminByEmail(email);
        LoginObject loginObject = new LoginObject();

        if (admin == null) {
            loginObject.setMessage("Admin with email " + email + " does not exist!");
            loginObject.setId(-1L);
            return ResponseEntity.badRequest().body(loginObject);
        }

        if (!admin.getPassword().equals(password)) {
            loginObject.setMessage("Incorrect password!");
            loginObject.setId(-1L);
            return ResponseEntity.badRequest().body(loginObject);
        }

        loginObject.setMessage("Admin logged in successfully!");
        loginObject.setId(admin.getId());
        return ResponseEntity.ok(loginObject);
    }

    public ResponseEntity<String> changePasswordByEmailAdmin(String email, String newPass, String oldPass) {
        Admin admin = adminRepository.findAdminByEmail(email);

        if (admin == null) {
            return ResponseEntity.badRequest().body("Admin with email " + email + " does not exist!");
        }

        if (!admin.getPassword().equals(oldPass)) {
            return ResponseEntity.badRequest().body("Incorrect password!");
        }

        if (admin.getPassword().equals(newPass)) {
            return ResponseEntity.badRequest().body("New password cannot be same as old password!");
        }

        admin.setPassword(newPass);
        adminRepository.save(admin);
        return ResponseEntity.ok("Password changed successfully!");
    }
}
