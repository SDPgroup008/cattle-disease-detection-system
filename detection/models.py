from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError

class Cattle(models.Model):
    tag_number = models.CharField(max_length=50, unique=True)
    breed = models.CharField(max_length=100)
    age = models.IntegerField()
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='cattle')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['tag_number']

    def __str__(self):
        return self.tag_number

class TestResult(models.Model):
    cattle = models.ForeignKey(Cattle, on_delete=models.CASCADE, related_name='test_results')
    image = models.ImageField(upload_to='uploads/')
    integrated_gradients_image = models.ImageField(upload_to='explainability/', null=True, blank=True)
    lime_image = models.ImageField(upload_to='explainability/', null=True, blank=True)
    occlusion_image = models.ImageField(upload_to='explainability/', null=True, blank=True)
    gradcam_image = models.ImageField(upload_to='explainability/', null=True, blank=True)
    foot_infection = models.BooleanField(default=False)
    mouth_infection = models.BooleanField(default=False)
    is_infected = models.BooleanField(default=False)
    is_healthy = models.BooleanField(default=True)
    tested_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='test_results')

    class Meta:
        ordering = ['-tested_at']

    def clean(self):
        if self.user != self.cattle.owner:
            raise ValidationError("The user creating the test result must be the owner of the cattle.")

    def save(self, *args, **kwargs):
        if not self.user:
            self.user = self.cattle.owner
        self.clean()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Test for {self.cattle.tag_number} on {self.tested_at}"