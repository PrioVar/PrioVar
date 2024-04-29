package com.bio.priovar.serializers;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.ser.std.StdSerializer;
import com.bio.priovar.models.Patient;

import java.io.IOException;

public class PatientLiteSerializer extends StdSerializer<Patient> {
    public PatientLiteSerializer() {
        this(null);
    }

    public PatientLiteSerializer(Class<Patient> t) {
        super(t);
    }

    @Override
    public void serialize(Patient patient, JsonGenerator jgen, SerializerProvider provider) throws IOException {
        jgen.writeStartObject();
        jgen.writeNumberField("id", patient.getId());
        jgen.writeStringField("name", patient.getName());
        jgen.writeStringField("sex", patient.getSex());
        jgen.writeNumberField("age", patient.getAge());
        jgen.writeEndObject();
    }
}
