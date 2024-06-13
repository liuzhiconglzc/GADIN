# Generated by Django 2.0.2 on 2023-01-22 11:15

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('book', '0058_auto_20230122_1857'),
    ]

    operations = [
        migrations.AlterField(
            model_name='borrowrecord',
            name='end_day',
            field=models.DateTimeField(default=datetime.datetime(2023, 1, 29, 11, 15, 12, 439810, tzinfo=utc)),
        ),
        migrations.AlterField(
            model_name='generalinfo',
            name='col_1',
            field=models.CharField(blank=True, max_length=128, null=True, verbose_name='属性1'),
        ),
        migrations.AlterField(
            model_name='generalinfo',
            name='col_10',
            field=models.CharField(blank=True, max_length=128, null=True, verbose_name='属性10'),
        ),
        migrations.AlterField(
            model_name='generalinfo',
            name='col_11',
            field=models.CharField(blank=True, max_length=128, null=True, verbose_name='属性11'),
        ),
        migrations.AlterField(
            model_name='generalinfo',
            name='col_12',
            field=models.CharField(blank=True, max_length=128, null=True, verbose_name='属性12'),
        ),
        migrations.AlterField(
            model_name='generalinfo',
            name='col_13',
            field=models.CharField(blank=True, max_length=128, null=True, verbose_name='属性13'),
        ),
        migrations.AlterField(
            model_name='generalinfo',
            name='col_14',
            field=models.CharField(blank=True, max_length=128, null=True, verbose_name='属性14'),
        ),
        migrations.AlterField(
            model_name='generalinfo',
            name='col_15',
            field=models.CharField(blank=True, max_length=128, null=True, verbose_name='属性15'),
        ),
        migrations.AlterField(
            model_name='generalinfo',
            name='col_16',
            field=models.CharField(blank=True, max_length=128, null=True, verbose_name='属性16'),
        ),
        migrations.AlterField(
            model_name='generalinfo',
            name='col_17',
            field=models.CharField(blank=True, max_length=128, null=True, verbose_name='属性17'),
        ),
        migrations.AlterField(
            model_name='generalinfo',
            name='col_18',
            field=models.CharField(blank=True, max_length=128, null=True, verbose_name='属性18'),
        ),
        migrations.AlterField(
            model_name='generalinfo',
            name='col_19',
            field=models.CharField(blank=True, max_length=128, null=True, verbose_name='属性19'),
        ),
        migrations.AlterField(
            model_name='generalinfo',
            name='col_2',
            field=models.CharField(blank=True, max_length=128, null=True, verbose_name='属性2'),
        ),
        migrations.AlterField(
            model_name='generalinfo',
            name='col_20',
            field=models.CharField(blank=True, max_length=128, null=True, verbose_name='属性20'),
        ),
        migrations.AlterField(
            model_name='generalinfo',
            name='col_3',
            field=models.CharField(blank=True, max_length=128, null=True, verbose_name='属性3'),
        ),
        migrations.AlterField(
            model_name='generalinfo',
            name='col_4',
            field=models.CharField(blank=True, max_length=128, null=True, verbose_name='属性4'),
        ),
        migrations.AlterField(
            model_name='generalinfo',
            name='col_5',
            field=models.CharField(blank=True, max_length=128, null=True, verbose_name='属性5'),
        ),
        migrations.AlterField(
            model_name='generalinfo',
            name='col_6',
            field=models.CharField(blank=True, max_length=128, null=True, verbose_name='属性6'),
        ),
        migrations.AlterField(
            model_name='generalinfo',
            name='col_7',
            field=models.CharField(blank=True, max_length=128, null=True, verbose_name='属性7'),
        ),
        migrations.AlterField(
            model_name='generalinfo',
            name='col_8',
            field=models.CharField(blank=True, max_length=128, null=True, verbose_name='属性8'),
        ),
        migrations.AlterField(
            model_name='generalinfo',
            name='col_9',
            field=models.CharField(blank=True, max_length=128, null=True, verbose_name='属性9'),
        ),
        migrations.AlterField(
            model_name='generalinfo',
            name='table_name',
            field=models.CharField(blank=True, max_length=128, null=True, verbose_name='表名'),
        ),
    ]
