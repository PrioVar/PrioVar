package com.bio.priovar.services;

import com.bio.priovar.models.Chat;
import com.bio.priovar.repositories.ChatRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class ChatService {

    private final ChatRepository chatRepository;

    @Autowired
    public ChatService(ChatRepository chatRepository) {
        this.chatRepository = chatRepository;
    }

    public List<Chat> getChatsByMedicalCenterId(Long medicalCenterId) {
        return chatRepository.findAllByMedicalCenterId(medicalCenterId);
    }
}
