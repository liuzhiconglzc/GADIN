# Generated by Django 2.0.2 on 2022-12-25 11:56

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('book', '0047_auto_20221223_1523'),
    ]

    operations = [
        migrations.AlterField(
            model_name='borrowrecord',
            name='end_day',
            field=models.DateTimeField(default=datetime.datetime(2023, 1, 1, 11, 56, 13, 481928, tzinfo=utc)),
        ),
    ]
