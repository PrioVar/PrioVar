package com.bio.priovar.serializers;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.ser.std.StdSerializer;
import com.bio.priovar.models.InformationRequest;

import java.io.IOException;

public class InformationRequestLiteSerializer  extends StdSerializer<InformationRequest> {
    public InformationRequestLiteSerializer() {
        this(null);
    }

    public InformationRequestLiteSerializer(Class<InformationRequest> t) {
        super(t);
    }

    @Override
    public void serialize(InformationRequest informationRequest, JsonGenerator jgen, SerializerProvider provider) throws IOException {
        jgen.writeStartObject();
        jgen.writeNumberField("id", informationRequest.getId());
        jgen.writeStringField("requestDescription", informationRequest.getRequestDescription());
        jgen.writeBooleanField("isPending", informationRequest.getIsPending());
        jgen.writeBooleanField("isApproved", informationRequest.getIsApproved());
        jgen.writeBooleanField("isRejected", informationRequest.getIsRejected());
        jgen.writeEndObject();
    }
}
