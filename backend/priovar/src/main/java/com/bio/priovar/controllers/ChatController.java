package com.bio.priovar.controllers;

import com.bio.priovar.models.Chat;
import com.bio.priovar.models.GraphChat;
import com.bio.priovar.services.ChatService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/chat")
@CrossOrigin
public class ChatController {

    private final ChatService chatService;

    @Autowired
    public ChatController(ChatService chatService) {
        this.chatService = chatService;
    }


    @GetMapping("/getChatsByMedicalCenterId/{medicalCenterId}")
    public List<Chat> getChatsByMedicalCenterId(@PathVariable("medicalCenterId") Long medicalCenterId) {
        return chatService.getChatsByMedicalCenterId(medicalCenterId);
    }

    @GetMapping("/getGraphChatsByMedicalCenterId/{medicalCenterId}")
    public List<GraphChat> getGraphChatsByMedicalCenterId(@PathVariable("medicalCenterId") Long medicalCenterId) {
        return chatService.getGraphChatsByMedicalCenterId(medicalCenterId);
    }
}
