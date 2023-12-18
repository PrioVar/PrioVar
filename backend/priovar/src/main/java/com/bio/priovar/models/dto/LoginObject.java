package com.bio.priovar.models.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class LoginObject {
    private Long id;
    private Long relatedId;
    private String message;
}
