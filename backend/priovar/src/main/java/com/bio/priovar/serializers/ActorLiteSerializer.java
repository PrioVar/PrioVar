package com.bio.priovar.serializers;

import com.bio.priovar.models.Actor;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.ser.std.StdSerializer;

import java.io.IOException;

public class ActorLiteSerializer extends StdSerializer<Actor> {
    public ActorLiteSerializer() {
        this(null);
    }

    public ActorLiteSerializer(Class<Actor> t) {
        super(t);
    }

    @Override
    public void serialize(Actor actor, JsonGenerator jgen, SerializerProvider provider) throws IOException {
        jgen.writeStartObject();
        jgen.writeNumberField("id", actor.getId());
        jgen.writeStringField("name", actor.getName());
        jgen.writeEndObject();
    }
}