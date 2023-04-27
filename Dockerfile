# Use an official Python runtime as the base image
FROM python:3.8-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the required files to the container
COPY api.py .
COPY auto_insurance_model.pkl .

# Install the required packages
RUN pip install flask
RUN pip install numpy
RUN pip install scikit-learn
RUN pip install pickle

# Specify the command to run the API
CMD ["python", "api.py"]

# Expose the API on port 80
EXPOSE 80
