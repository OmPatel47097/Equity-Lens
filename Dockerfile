# Use an appropriate base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Expose the port Streamlit will run on
EXPOSE 80

# Create Streamlit configuration directory and copy configuration files
RUN mkdir -p ~/.streamlit
COPY .streamlit/config.toml ~/.streamlit/config.toml
COPY .streamlit/credentials.toml ~/.streamlit/credentials.toml

# Run the Streamlit app
CMD ["streamlit", "run", "prototypes/blpo.py"]
