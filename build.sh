#!/usr/bin/env bash
# exit on error
set -o errexit

echo "🚀 Starting build process..."

# Install Python dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements_render.txt

# Download the AI model
echo "🤖 Downloading AI model..."
python detection/model_downloader.py

# Collect static files
echo "📁 Collecting static files..."
python manage_render.py collectstatic --no-input

# Run database migrations
echo "🗄️ Running database migrations..."
python manage_render.py migrate

# Create superuser if it doesn't exist
echo "👤 Setting up admin user..."
python manage_render.py shell << EOF
from django.contrib.auth.models import User
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@example.com', 'admin123')
    print('✅ Superuser created: admin/admin123')
else:
    print('✅ Superuser already exists')
EOF

echo "✅ Build completed successfully!"
