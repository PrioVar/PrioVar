# Use the official Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the wait-for-it.sh script and your application files into the working directory
COPY wait-for-it.sh .
COPY . .

# Make wait-for-it.sh executable
RUN chmod +x wait-for-it.sh

# Specify the port the app runs on
EXPOSE 5000

# Use an entrypoint script to wait for Neo4j with a timeout of 30 seconds
ENTRYPOINT ["./wait-for-it.sh", "neo4j:7687", "--timeout=30", "--"]

# Command to run the application after the entrypoint
CMD ["flask", "run", "--host=0.0.0.0"]