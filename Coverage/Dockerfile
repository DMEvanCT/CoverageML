# Use an official Python runtime as the base image
FROM python:3.8-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the required files to the container
COPY api.py .
COPY auto_insurance_model.pkl .
COPY ../requirements.txt .

# Install the required packages
RUN pip install -r requirements.txt

# Specify the command to run the API
CMD ["python", "api.py"]

# Expose the API on port 8080
EXPOSE 5000
