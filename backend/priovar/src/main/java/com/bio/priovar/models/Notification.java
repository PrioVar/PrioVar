package com.bio.priovar.models;

import com.bio.priovar.serializers.ActorLiteSerializer;
import com.bio.priovar.serializers.InformationRequestLiteSerializer;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.data.neo4j.core.schema.GeneratedValue;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.Node;
import org.springframework.data.neo4j.core.schema.Relationship;


import java.time.OffsetDateTime;

@Getter
@Setter
@NoArgsConstructor
@Node("Notification")
public class Notification {
    @Id
    @GeneratedValue
    private Long id;
    private String type; //Request, Response
    private String notification;
    private OffsetDateTime sendAt;
    private String appendix; //Holds any extra information that needs to be sent with the notification
    private Boolean isRead;


    @Relationship(type = "NOTIFIED_BY", direction = Relationship.Direction.OUTGOING)
    @JsonSerialize(using = ActorLiteSerializer.class)
    private Actor sender;

    @Relationship(type = "NOTIFIED_TO", direction = Relationship.Direction.OUTGOING)
    @JsonSerialize(using = ActorLiteSerializer.class)
    private Actor receiver;

    @Relationship(type = "CARRIES", direction = Relationship.Direction.OUTGOING)
    @JsonSerialize(using = InformationRequestLiteSerializer.class)
    private InformationRequest informationRequest;


}
