# Generated by Django 2.0.2 on 2023-01-17 01:56

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('book', '0051_auto_20230117_0953'),
    ]

    operations = [
        migrations.AlterField(
            model_name='borrowrecord',
            name='end_day',
            field=models.DateTimeField(default=datetime.datetime(2023, 1, 24, 1, 56, 1, 487830, tzinfo=utc)),
        ),
        migrations.AlterModelTable(
            name='filledparameter',
            table='filledparameter',
        ),
    ]
