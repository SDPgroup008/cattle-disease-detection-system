"""
WSGI config for cattle_disease project on Render.
"""

import os
from django.core.wsgi import get_wsgi_application

# Use render settings for production
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cattle_disease.render_settings')

application = get_wsgi_application()
