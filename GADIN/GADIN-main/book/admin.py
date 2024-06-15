from django.contrib import admin

# hmf 后台admin配置文件
# Register your models here.
from.models import Category,Publisher,Profile,Member,BorrowRecord

from .forms import BorrowRecordCreateForm

# @admin.register(Member)
# class MemberAdmin(AjaxSelectAdmin):
#     form = BorrowRecordCreateForm

admin.site.register(Member)
admin.site.register(BorrowRecord)