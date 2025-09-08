# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir: Disables the cache, which reduces the image size
# --trusted-host pypi.python.org: Can help avoid SSL issues in some networks
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# Copy the application source code and models to the container
COPY ./src /app/src
COPY ./models /app/models

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define the command to run your app using uvicorn
# It will run the 'app' instance from the 'predict_api.py' file in the 'src' module
CMD ["uvicorn", "src.predict_api:app", "--host", "0.0.0.0", "--port", "8000"]