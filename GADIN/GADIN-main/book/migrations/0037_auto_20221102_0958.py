# Generated by Django 2.0.2 on 2022-11-02 01:58

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('book', '0036_auto_20221101_1807'),
    ]

    operations = [
        migrations.AlterField(
            model_name='borrowrecord',
            name='end_day',
            field=models.DateTimeField(default=datetime.datetime(2022, 11, 9, 1, 58, 1, 373885, tzinfo=utc)),
        ),
    ]