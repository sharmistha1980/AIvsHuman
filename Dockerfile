# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /code

# Copy the current directory contents into the container at /code
COPY . .

# Install any needed packages specified in requirements.txt
# We increase timeout because PyTorch/Transformers are large downloads
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Make port 7860 available to the world outside this container
# Hugging Face Spaces expects apps to run on port 7860
EXPOSE 7860

# Define environment variable
ENV FLASK_APP=app.py

# Run app.py when the container launches
CMD ["flask", "run", "--host=0.0.0.0", "--port=7860"]