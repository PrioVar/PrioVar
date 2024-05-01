package com.bio.priovar.services;

import com.bio.priovar.models.Chat;
import com.bio.priovar.models.GraphChat;
import com.bio.priovar.repositories.ChatRepository;
import com.bio.priovar.repositories.GraphChatRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Comparator;
import java.util.List;

@Service
public class ChatService {

    private final ChatRepository chatRepository;
    private final GraphChatRepository graphChatRepository;

    @Autowired
    public ChatService(ChatRepository chatRepository, GraphChatRepository graphChatRepository) {
        this.chatRepository = chatRepository;
        this.graphChatRepository = graphChatRepository;
    }

    public List<Chat> getChatsByMedicalCenterId(Long medicalCenterId) {
        List<Chat> chats = chatRepository.findAllByMedicalCenterId(medicalCenterId);

        // sort the chats by String timestamp, oldest first
        chats.sort(Comparator.comparing(Chat::getTimestamp));

        return chats;
    }

    public List<GraphChat> getGraphChatsByMedicalCenterId(Long medicalCenterId) {
        List<GraphChat> graphChats = graphChatRepository.findAllByMedicalCenterId(medicalCenterId);

        // sort the graph chats by String timestamp, oldest first
        graphChats.sort(Comparator.comparing(GraphChat::getTimestamp));

        return graphChats;
    }
}
