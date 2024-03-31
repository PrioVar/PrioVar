# Use a Maven base image for building the app
FROM maven:3.9.6-eclipse-temurin-17 as build-stage

# Set the working directory
WORKDIR /app

# Copy the pom.xml and source code
COPY pom.xml ./
COPY src ./src

# Build the application
RUN mvn clean package

# Use OpenJDK 17 for running the app
FROM eclipse-temurin:17-jdk as production-stage

# Set the working directory
WORKDIR /app

# Copy the JAR file from the build stage(CHANGE ACCORDING TO YOUR JAR FILE NAME and PATH)
COPY ./target/priovar-0.0.1-SNAPSHOT.jar app.jar

# Expose the port the app runs on
EXPOSE 8080

# Run the app
CMD ["java", "-jar", "app.jar"]