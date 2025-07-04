# Generated by Django 4.2 on 2025-05-24 14:25

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("detection", "0001_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="cattle",
            name="owner",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="cattle",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AlterField(
            model_name="testresult",
            name="cattle",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="test_results",
                to="detection.cattle",
            ),
        ),
        migrations.AlterField(
            model_name="testresult",
            name="user",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="test_results",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
    ]
