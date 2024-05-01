package com.bio.priovar.models;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.data.neo4j.core.schema.GeneratedValue;
import org.springframework.data.neo4j.core.schema.Id;

@Getter
@Setter
@NoArgsConstructor
public class Actor {

    @Id
    @GeneratedValue
    private Long id;

    private String name;
    private String email;
    private String password;
}
