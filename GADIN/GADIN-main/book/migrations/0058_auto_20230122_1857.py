# Generated by Django 2.0.2 on 2023-01-22 10:57

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('book', '0057_auto_20230122_1607'),
    ]

    operations = [
        migrations.AlterField(
            model_name='borrowrecord',
            name='end_day',
            field=models.DateTimeField(default=datetime.datetime(2023, 1, 29, 10, 57, 44, 217159, tzinfo=utc)),
        ),
    ]