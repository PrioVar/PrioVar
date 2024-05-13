# PRIOVAR
PrioVar is a cutting-edge bioinformatics application tailored for genetic research centers. It boasts a suite of advanced features including variant prioritization, leveraging a novel machine learning scoring system, and comprehensive variant annotation. PrioVar integrates unique LLM features; one utilizes RAG for searching the PubMed database, while another facilitates knowledge graph retrieval from the Neo4j database. This robust platform is engineered to streamline data analysis processes, utilizing state-of-the-art container technology and a potent mix of backend services, making it a powerhouse for genetic data management and research.

## 🐳 Docker Installation
To set up PRIOVAR using Docker, follow these steps to build and run the services:

### Building and Running the Services
In the project's root directory, execute the following commands to build and start the services:

```bash
docker-compose build
docker-compose up
```
### Accessing the Application

Once the services are up, you can access the PRIOVAR platform through your browser:

- **Frontend**: [http://localhost:3000](http://localhost:3000)
- **Flask API**: [http://localhost:5000](http://localhost:5000)
- **Spring Boot**: [http://localhost:8080](http://localhost:8080)
- **Neo4j Database**: Bolt port at `7687`

### Stopping the Services

To stop and remove all running services, use the following command:

```bash
docker-compose down
```

## 🛠️ Manual Installation
For manual installation, follow the step-by-step guide below to set up each component of the PRIOVAR system.

### Frontend Setup

1. **Navigate to `frontend_cs491_org`**:
   - Ensure all necessary modules are installed:
     ```bash
     yarn install
     ```
   - This might take a few minutes. Once the installation is complete, start the frontend server:
     ```bash
     yarn start
     ```

   - The frontend will now be running at [http://localhost:3000](http://localhost:3000).
### Database Setup
2. **Prepare the Neo4j Graph Database**:
   - **Ensure Neo4j is Running**: Before proceeding with the following steps, verify that the Neo4j Graph Database is up and operational.
   - **Update Configuration Settings**:
     - In the Flask application, update the username and password information in `config.py`.
     - In the Spring Boot application, modify the username and password details in `application.properties`.
### Backend Setup
3. **Start the Spring Boot Server**:
   - Navigate to the `backend` directory and follow the instructions specific to Spring Boot to get the server running.
4. **Set Up the Flask Server + Python Environment**:
   - Navigate to the `flask` directory.
   - Create a Python virtual environment and activate it:
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```
   - Install the necessary libraries:
     ```bash
     pip install -r requirements.txt
     ```
   - Run `app.py` to start the Flask application:
     ```bash
     python app.py
     ```

   - After executing this step, you can log in to the system using any actor available on the Neo4j Graph Database.
