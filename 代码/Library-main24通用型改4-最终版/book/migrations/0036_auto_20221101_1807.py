# Generated by Django 2.0.2 on 2022-11-01 10:07

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('book', '0035_auto_20221101_1759'),
    ]

    operations = [
        migrations.AlterField(
            model_name='borrowrecord',
            name='end_day',
            field=models.DateTimeField(default=datetime.datetime(2022, 11, 8, 10, 7, 24, 459685, tzinfo=utc)),
        ),
    ]
