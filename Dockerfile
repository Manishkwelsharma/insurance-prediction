# Use Python 3.11.9 as the base image
FROM python:3.11.9

# Set the working directory inside the container (use Unix-style paths)
WORKDIR  D:\projects\orison_tech\Regression

# Copy all project files into the container
COPY . .

# Install required dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the Streamlit app will run on
EXPOSE 8502

# Command to run the Streamlit app
CMD ["streamlit", "run", "model.py", "--server.port=8502", "--server.address=0.0.0.0"]
