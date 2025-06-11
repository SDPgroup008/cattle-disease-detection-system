from django.urls import path
from . import views

urlpatterns = [
    path('accounts/signup/', views.signup, name='signup'),
    path('', views.dashboard, name='dashboard'),
    path('api/dashboard-data/', views.dashboard_data, name='dashboard_data'),
    path('upload/', views.upload_image, name='upload_image'),
    path('results/<int:test_id>/', views.results, name='results'),
    path('history/', views.history, name='history'),
    path('register-cattle/', views.register_cattle, name='register_cattle'),
    path('register/', views.register_user, name='register_user'),
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('download-guide/', views.download_user_guide, name='download_user_guide'),
    path('help/', views.help_center, name='help_center'),
]
