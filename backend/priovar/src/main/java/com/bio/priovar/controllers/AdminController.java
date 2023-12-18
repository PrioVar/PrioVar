package com.bio.priovar.controllers;

import com.bio.priovar.models.Admin;
import com.bio.priovar.models.dto.LoginObject;
import com.bio.priovar.services.AdminService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/admin")
@CrossOrigin
public class AdminController {
    private final AdminService adminService;

    @Autowired
    public AdminController(AdminService adminService) {
        this.adminService = adminService;
    }

    @PostMapping("/add")
    public ResponseEntity<String> addAdmin(@RequestBody Admin admin) {
        System.out.println(admin.getEmail());
        System.out.println(admin.getPassword());
        return adminService.addAdmin(admin);
    }

    @PostMapping("/login")
    public ResponseEntity<LoginObject> loginAdmin(@RequestParam String email, @RequestParam String password) {
        return adminService.loginAdmin(email,password);
    }

    @PatchMapping("/changePassword")
    public ResponseEntity<String> changePasswordAdmin(@RequestParam String email, @RequestParam String newPass, @RequestParam String oldPass) {
        return adminService.changePasswordByEmailAdmin(email, newPass, oldPass);
    }
}
