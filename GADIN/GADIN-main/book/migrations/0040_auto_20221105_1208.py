# Generated by Django 2.0.2 on 2022-11-05 04:08

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('book', '0039_auto_20221105_0913'),
    ]

    operations = [
        migrations.AlterField(
            model_name='borrowrecord',
            name='end_day',
            field=models.DateTimeField(default=datetime.datetime(2022, 11, 12, 4, 8, 0, 479103, tzinfo=utc)),
        ),
    ]