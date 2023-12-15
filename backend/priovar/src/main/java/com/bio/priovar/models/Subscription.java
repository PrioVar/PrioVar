package com.bio.priovar.models;

import lombok.Getter;

@Getter
public enum Subscription {
    // 1. Basic
    // 2. Premium
    // 3. Enterprise
    // Basic: $10, 10 analyses
    // Premium: $20, 20 analyses
    // Enterprise: $30, 30 analyses

    BASIC("Basic", 10, 10),
    PREMIUM("Premium", 20, 20),
    ENTERPRISE("Enterprise", 30, 30);

    private final String name;
    private final int price;
    private final int analyses;

    Subscription(String name, int price, int analyses) {
        this.name = name;
        this.price = price;
        this.analyses = analyses;
    }
}
