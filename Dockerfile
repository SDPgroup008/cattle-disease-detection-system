# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        postgresql-client \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_render.txt .
RUN pip install --no-cache-dir -r requirements_render.txt

# Copy project
COPY . .

# Make build script executable
RUN chmod +x build.sh

# Collect static files
RUN python manage.py collectstatic --no-input --settings=cattle_disease.render_settings

# Expose port
EXPOSE 8000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "cattle_disease.wsgi_render:application"]
