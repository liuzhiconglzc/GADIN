# Generated by Django 2.0.2 on 2023-03-13 02:23

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('book', '0062_auto_20230312_2027'),
    ]

    operations = [
        migrations.AlterField(
            model_name='borrowrecord',
            name='end_day',
            field=models.DateTimeField(default=datetime.datetime(2023, 3, 20, 2, 23, 20, 97820, tzinfo=utc)),
        ),
    ]
