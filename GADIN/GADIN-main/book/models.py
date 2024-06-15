from django.db import models
from django.contrib.auth.models import AbstractUser
from django.dispatch import receiver
from django.db.models.signals import post_save
from django.conf import settings
from django.contrib.auth.models import User
from django.urls import reverse
from django.utils import timezone
# from phonenumber_field.modelfields import PhoneNumberField
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import uuid
from PIL import Image

# todo hmf 数据库模型文件
# todo model文件，类似创建数据库表 或 连接数据库表

# todo 通用属性的内容，相当于常量，类似性别
# todo 写完数据库的东西，要run一下manage.py文件？
BOOK_STATUS =(
    (0, "On loan"),
    (1, "In Stock"),
)

FLOOR =(
    (1, "1st"),
    (2, "2nd"),
    (3, "3rd"),
)

OPERATION_TYPE =(
    ("success", "Create"),
    ("warning","Update"),
    ("danger","Delete"),
    ("info",'Close')
)

GENDER=(
    ("m","Male"),
    ("f","Female"),
)

BORROW_RECORD_STATUS=(
    (0,'Open'),
    (1,'Closed')
)

class Category(models.Model):
    
    name = models.CharField(max_length=50, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.name
    def get_absolute_url(self): 
        return reverse('category_list')

    # class Meta:
    #     db_table='category'

class Publisher(models.Model):
    
    name = models.CharField(max_length=50, blank=True)
    city = models.CharField(max_length=50, blank=True)
    contact = models.EmailField(max_length=50,blank=True)
    # todo 上面的邮箱定义，在数据库中是varcahr形式
    # created_at = models.DateTimeField(auto_now_add=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_by=models.CharField(max_length=20,default='yaozeliang')
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    def get_absolute_url(self): 
        return reverse('publisher_list')

class Book(models.Model):
    author = models.CharField("Author",max_length=20)
    title = models.CharField('Title',max_length=100)
    description = models.TextField()
    created_at = models.DateTimeField('Created Time',default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    total_borrow_times = models.PositiveIntegerField(default=0)
    quantity = models.PositiveIntegerField(default=10)
    category = models.ForeignKey(
        Category,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='category'
    )
    # todo 注意主键，外键
    publisher=models.ForeignKey(
        Publisher,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='publisher'
    )

    status=models.IntegerField(choices=BOOK_STATUS,default=1)
    floor_number=models.IntegerField(choices=FLOOR,default=1)
    bookshelf_number=models.CharField('Bookshelf Number',max_length=10,default='0001')
    updated_by=models.CharField(max_length=20,default='yaozeliang')

    def get_absolute_url(self): 
        return reverse('book_list')
    
    def __str__(self):
        return self.title

class UserActivity(models.Model):
    created_by=models.CharField(default="",max_length=20)
    created_at =models.DateTimeField(auto_now_add=True)
    operation_type=models.CharField(choices=OPERATION_TYPE,default="success",max_length=20)
    target_model = models.CharField(default="",max_length=20)
    detail = models.CharField(default="",max_length=50)

    def get_absolute_url(self): 
        return reverse('user_activity_list')

class Member(models.Model):
    name = models.CharField(max_length=50, blank=False)
    age = models.PositiveIntegerField(default=20)
    gender = models.CharField(max_length=10,choices=GENDER,default='m')

    city = models.CharField(max_length=20, blank=False)
    email = models.EmailField(max_length=50,blank=True)
    phone_number = models.CharField(max_length=30,blank=False)

    created_at= models.DateTimeField(default=timezone.now)
    created_by = models.CharField(max_length=20,default="")
    updated_by = models.CharField(max_length=20,default="")
    updated_at = models.DateTimeField(auto_now=True)

    card_id = models.UUIDField(unique=True, default=uuid.uuid4, editable=False)
    card_number = models.CharField(max_length=8,default="")
    expired_at = models.DateTimeField(default=timezone.now)

    def get_absolute_url(self): 
        return reverse('member_list')
    
    def save(self, *args, **kwargs):
        self.card_number = str(self.card_id)[:8]
        self.expired_at = timezone.now()+relativedelta(years=1)
        return super(Member, self).save(*args, **kwargs)

    def __str__(self):
        return self.name


# UserProfile
class Profile(models.Model):
    user = models.OneToOneField(User,null=True,on_delete=models.CASCADE)
    bio = models.TextField()
    profile_pic = models.ImageField(upload_to="profile/%Y%m%d/", blank=True,null=True)
    phone_number = models.CharField(max_length=30,blank=True)
    email = models.EmailField(max_length=50,blank=True)

    def save(self, *args, **kwargs):
        # 调用原有的 save() 的功能
        profile = super(Profile, self).save(*args, **kwargs)

        # 固定宽度缩放图片大小
        if self.profile_pic and not kwargs.get('update_fields'):
            image = Image.open(self.profile_pic)
            (x, y) = image.size
            new_x = 400
            new_y = int(new_x * (y / x))
            resized_image = image.resize((new_x, new_y), Image.ANTIALIAS)
            resized_image.save(self.profile_pic.path)

        return profile

    def __str__(self):
        return str(self.user)

    def get_absolute_url(self): 
        return reverse('home')


# Borrow Record

class BorrowRecord(models.Model):

    borrower = models.CharField(blank=False,max_length=20)
    borrower_card = models.CharField(max_length=8,blank=True)
    borrower_email = models.EmailField(max_length=50,blank=True)
    borrower_phone_number  = models.CharField(max_length=30,blank=True)
    book = models.CharField(blank=False,max_length=20)
    quantity = models.PositiveIntegerField(default=1)

    start_day = models.DateTimeField(default=timezone.now)
    end_day = models.DateTimeField(default=timezone.now()+timedelta(days=7))
    periode = models.PositiveIntegerField(default=0)

    open_or_close = models.IntegerField(choices=BORROW_RECORD_STATUS,default=0)
    delay_days = models.IntegerField(default=0)
    final_status = models.CharField(max_length=10,default="Unknown")

    created_at= models.DateTimeField(default=timezone.now)
    created_by = models.CharField(max_length=20,blank=True)
    closed_by = models.CharField(max_length=20,default="")
    closed_at = models.DateTimeField(auto_now=True)

    @property
    def return_status(self):
        if self.final_status!="Unknown":
            return self.final_status
        elif self.end_day.replace(tzinfo=None) > datetime.now()-timedelta(hours=24):
            return 'On time'
        else:
            return 'Delayed'

    @property
    def get_delay_number_days(self):
        
        if self.delay_days!=0:
            return self.delay_days
        elif self.return_status=='Delayed':
            return (datetime.now()-self.end_day.replace(tzinfo=None)).days
        else:
            return 0


    def get_absolute_url(self): 
        return reverse('record_list')

    def __str__(self):
        return self.borrower+"->"+self.book
    
    def save(self, *args, **kwargs):
        # profile = super(Profile, self).save(*args, **kwargs)
        self.periode =(self.end_day - self.start_day).days+1
        return super(BorrowRecord, self).save(*args, **kwargs)

# todo 用戶信息模型
class UserInfo(models.Model):
    username = models.CharField('账户名', max_length=32)
    password = models.CharField('密码', max_length=32)
    gender = models.CharField('性别', max_length=32)
    # admin = models.CharField('管理员权限', max_length=32)
    addr = models.CharField('地址', max_length=64)
    # todo 在数据库中仍是 verchar 格式
    email = models.EmailField('邮箱')
    # todo 手机号的位数多
    phone = models.BigIntegerField('联系电话')

    # todo 作用不知
    def __str__(self):
        return f'用户名：{self.username}'

    # todo 这个关系到views的一些方法的成功，类似完成任务后的跳转界面
    def get_absolute_url(self):
        return reverse('user_info')

    class Meta:
        db_table = "userinfo"


# todo 测试的金融数据，资产负债信息
class BankInfo(models.Model):
    zcfz_id = models.CharField('ID', max_length=128)
    company_name = models.CharField('公司名', max_length=128, null=True)
    type = models.CharField('类型', max_length=128, null=True)
    bbd_url = models.CharField('数据来源', max_length=128, null=True)
    bbd_type = models.CharField('数据类型', max_length=128, null=True)
    cash_central_bank_funds = models.CharField('现金及存放中央银行款项(元)', max_length=128, null=True)
    sellable_assetset = models.CharField('可供出售金融资产(元)', max_length=128, null=True)
    risk_preparation = models.CharField('一般风险准备(元)', max_length=128, null=True)
    interest_payable = models.CharField('应付利息(元)', max_length=128, null=True)
    paid_in_capital = models.CharField('实收资本', max_length=128, null=True)
    fixed_assets = models.CharField('固定资产', max_length=128, null=True)
    total_assets = models.CharField('资产总计', max_length=128, null=True)
    capital_reserves = models.CharField('资本公积', max_length=128, null=True)

    # todo 作用不知
    def __str__(self):
        return f'公司名：....股份有限公司'

    def get_absolute_url(self):
        return reverse('upload_bank')

    class Meta:
        db_table = "bankinfo"

# todo 配合冗余属性的 假删除 + 计算缺失率的
class DelConInfo(models.Model):
    del_name = models.CharField('删除属性', max_length=128, null=True)
    del_status = models.CharField('布尔值', max_length=128, null=True)

    # 保存缺失率
    loss_rate = models.CharField('缺失率', max_length=128, null=True)
    loss_rate_status = models.CharField('计算状态', max_length=128, null=True)

    # todo 作用不知
    def __str__(self):
        return f'公司名：....'

    class Meta:
        db_table = "delconinfo"




# todo 测试的金融数据，资产负债信息---用于删除属性的标记（包含所有属性）
class BankInfoDelMark(models.Model):
    # zcfz_id = models.CharField('ID', max_length=128)
    company_name = models.CharField('公司名', max_length=128, null=True)
    type = models.CharField('类型', max_length=128, null=True)
    bbd_url = models.CharField('数据来源', max_length=128, null=True)
    bbd_type = models.CharField('数据类型', max_length=128, null=True)
    cash_central_bank_funds = models.CharField('现金及存放中央银行款项(元)', max_length=128, null=True)
    sellable_assetset = models.CharField('可供出售金融资产(元)', max_length=128, null=True)
    risk_preparation = models.CharField('一般风险准备(元)', max_length=128, null=True)
    interest_payable = models.CharField('应付利息(元)', max_length=128, null=True)
    paid_in_capital = models.CharField('实收资本', max_length=128, null=True)
    fixed_assets = models.CharField('固定资产', max_length=128, null=True)
    total_assets = models.CharField('资产总计', max_length=128, null=True)
    capital_reserves = models.CharField('资本公积', max_length=128, null=True)

    # todo 作用不知
    def __str__(self):
        return f'公司名：....股份有限公司'

    class Meta:
        db_table = "bankinfodelmark"


# todo 保存模型参数
class ModelParameter(models.Model):
    # zcfz_id = models.CharField('ID', max_length=128)
    model_name = models.CharField('模型名称', max_length=128, null=True)
    model_G = models.CharField('生成器参数', max_length=128, null=True)
    model_status = models.CharField('模型训练状态', max_length=128, null=True)

    # todo 作用不知
    def __str__(self):
        return f'模型参数'

    class Meta:
        db_table = "modelparameter"


# todo 填补后的数据表
class BankInfoFilled(models.Model):
    zcfz_id = models.CharField('ID', max_length=128)
    company_name = models.CharField('公司名', max_length=128, null=True)
    type = models.CharField('类型', max_length=128, null=True)
    bbd_url = models.CharField('数据来源', max_length=128, null=True)
    bbd_type = models.CharField('数据类型', max_length=128, null=True)
    cash_central_bank_funds = models.CharField('现金及存放中央银行款项(元)', max_length=128, null=True)
    sellable_assetset = models.CharField('可供出售金融资产(元)', max_length=128, null=True)
    risk_preparation = models.CharField('一般风险准备(元)', max_length=128, null=True)
    interest_payable = models.CharField('应付利息(元)', max_length=128, null=True)
    paid_in_capital = models.CharField('实收资本', max_length=128, null=True)
    fixed_assets = models.CharField('固定资产', max_length=128, null=True)
    total_assets = models.CharField('资产总计', max_length=128, null=True)
    capital_reserves = models.CharField('资本公积', max_length=128, null=True)

    # todo 作用不知
    def __str__(self):
        return f'公司名：....股份有限公司'

    class Meta:
        db_table = "bankinfofilled"

# todo 填补效果表
class FilledParameter(models.Model):
    # zcfz_id = models.CharField('ID', max_length=128)
    table_name = models.CharField('表名', max_length=128, null=True)
    rmse = models.CharField('误差值', max_length=128, null=True)
    status = models.CharField('备注', max_length=128, null=True)

    class Meta:
        db_table = "filledparameter"


# todo 参数信息表
class ParameterInfo(models.Model):
    fill_name = models.CharField('表名', max_length=128, null=True)
    bz = models.CharField('备注', max_length=128, null=True)
    # todo 表名没APP前缀
    class Meta:
        db_table = "parameterinfo"


# todo 第二个测试表 yhzx
class YhzxInfo(models.Model):
    yhzx_id = models.CharField('ID', max_length=128)
    company_name = models.CharField('公司名', max_length=128, null=True)
    capital_adequacy_ratio = models.CharField('资本充足率(%)', max_length=128, null=True)
    provision_coverage = models.CharField('拨备覆盖率(%)', max_length=128, null=True)
    total_deposit = models.CharField('存款总额（%）', max_length=128, null=True)
    total_loan = models.CharField('贷款总额（%）', max_length=128, null=True)
    non_interest_income = models.CharField('非利息收入（元）', max_length=128, null=True)
    net_interest_margin = models.CharField('净息差(%)', max_length=128, null=True)
    # todo 作用不知
    def __str__(self):
        return f'公司名：....股份有限公司'

    class Meta:
        db_table = "yhzxinfo"

    def get_absolute_url(self):
        return reverse('upload_bank')

# todo 第二个测试表--删除标记表 yhzx
class YhzxInfoDelMark(models.Model):
    # yhzx_id = models.CharField('ID', max_length=128)
    company_name = models.CharField('公司名', max_length=128, null=True)
    capital_adequacy_ratio = models.CharField('资本充足率(%)', max_length=128, null=True)
    provision_coverage = models.CharField('拨备覆盖率(%)', max_length=128, null=True)
    total_deposit = models.CharField('存款总额（%）', max_length=128, null=True)
    total_loan = models.CharField('贷款总额（%）', max_length=128, null=True)
    non_interest_income = models.CharField('非利息收入（元）', max_length=128, null=True)
    net_interest_margin = models.CharField('净息差(%)', max_length=128, null=True)
    # todo 作用不知
    def __str__(self):
        return f'公司名：....股份有限公司'

    class Meta:
        db_table = "yhzxinfodelmark"

# todo 第二个测试表--填补后的表 yhzx
class YhzxInfoFilled(models.Model):
    yhzx_id = models.CharField('ID', max_length=128)
    company_name = models.CharField('公司名', max_length=128, null=True)
    capital_adequacy_ratio = models.CharField('资本充足率(%)', max_length=128, null=True)
    provision_coverage = models.CharField('拨备覆盖率(%)', max_length=128, null=True)
    total_deposit = models.CharField('存款总额（%）', max_length=128, null=True)
    total_loan = models.CharField('贷款总额（%）', max_length=128, null=True)
    non_interest_income = models.CharField('非利息收入（元）', max_length=128, null=True)
    net_interest_margin = models.CharField('净息差(%)', max_length=128, null=True)
    # todo 作用不知
    def __str__(self):
        return f'公司名：....股份有限公司'

    class Meta:
        db_table = "yhzxinfofilled"



# todo 通用表
class GeneralInfo(models.Model):
    table_name = models.CharField('表名', max_length=128, null=True, blank=True)
    col_1 = models.CharField('属性1', max_length=128, null=True, blank=True)
    col_2 = models.CharField('属性2', max_length=128, null=True, blank=True)
    col_3 = models.CharField('属性3', max_length=128, null=True, blank=True)
    col_4 = models.CharField('属性4', max_length=128, null=True, blank=True)
    col_5 = models.CharField('属性5', max_length=128, null=True, blank=True)
    col_6 = models.CharField('属性6', max_length=128, null=True, blank=True)
    col_7 = models.CharField('属性7', max_length=128, null=True, blank=True)
    col_8 = models.CharField('属性8', max_length=128, null=True, blank=True)
    col_9 = models.CharField('属性9', max_length=128, null=True, blank=True)
    col_10 = models.CharField('属性10', max_length=128, null=True, blank=True)
    col_11 = models.CharField('属性11', max_length=128, null=True, blank=True)
    col_12 = models.CharField('属性12', max_length=128, null=True, blank=True)
    col_13 = models.CharField('属性13', max_length=128, null=True, blank=True)
    col_14 = models.CharField('属性14', max_length=128, null=True, blank=True)
    col_15 = models.CharField('属性15', max_length=128, null=True, blank=True)
    col_16 = models.CharField('属性16', max_length=128, null=True, blank=True)
    col_17 = models.CharField('属性17', max_length=128, null=True, blank=True)
    col_18 = models.CharField('属性18', max_length=128, null=True, blank=True)
    col_19 = models.CharField('属性19', max_length=128, null=True, blank=True)
    col_20 = models.CharField('属性20', max_length=128, null=True, blank=True)
    # todo 作用不知
    def __str__(self):
        return f'通用数据表'

    class Meta:
        db_table = "generalinfo"

    def get_absolute_url(self):
        return reverse('upload_bank')


# todo 通用表-表信息
class GeneralTableInfo(models.Model):
    table_name = models.CharField('表名', max_length=128, null=True)
    col_number = models.IntegerField('属性个数', null=True)
    col_1 = models.CharField('属性1', max_length=128, null=True)
    col_2 = models.CharField('属性2', max_length=128, null=True)
    col_3 = models.CharField('属性3', max_length=128, null=True)
    col_4 = models.CharField('属性4', max_length=128, null=True)
    col_5 = models.CharField('属性5', max_length=128, null=True)
    col_6 = models.CharField('属性6', max_length=128, null=True)
    col_7 = models.CharField('属性7', max_length=128, null=True)
    col_8 = models.CharField('属性8', max_length=128, null=True)
    col_9 = models.CharField('属性9', max_length=128, null=True)
    col_10 = models.CharField('属性10', max_length=128, null=True)
    col_11 = models.CharField('属性11', max_length=128, null=True)
    col_12 = models.CharField('属性12', max_length=128, null=True)
    col_13 = models.CharField('属性13', max_length=128, null=True)
    col_14 = models.CharField('属性14', max_length=128, null=True)
    col_15 = models.CharField('属性15', max_length=128, null=True)
    col_16 = models.CharField('属性16', max_length=128, null=True)
    col_17 = models.CharField('属性17', max_length=128, null=True)
    col_18 = models.CharField('属性18', max_length=128, null=True)
    col_19 = models.CharField('属性19', max_length=128, null=True)
    col_20 = models.CharField('属性20', max_length=128, null=True)
    # todo 作用不知
    def __str__(self):
        return f'公司名：....股份有限公司'

    class Meta:
        db_table = "generaltableinfo"

# todo 通用表-标记删除属性
class GeneralMaskInfo(models.Model):
    table_name = models.CharField('表名', max_length=128, null=True)
    col_1 = models.CharField('属性1', max_length=128, null=True)
    col_2 = models.CharField('属性2', max_length=128, null=True)
    col_3 = models.CharField('属性3', max_length=128, null=True)
    col_4 = models.CharField('属性4', max_length=128, null=True)
    col_5 = models.CharField('属性5', max_length=128, null=True)
    col_6 = models.CharField('属性6', max_length=128, null=True)
    col_7 = models.CharField('属性7', max_length=128, null=True)
    col_8 = models.CharField('属性8', max_length=128, null=True)
    col_9 = models.CharField('属性9', max_length=128, null=True)
    col_10 = models.CharField('属性10', max_length=128, null=True)
    col_11 = models.CharField('属性11', max_length=128, null=True)
    col_12 = models.CharField('属性12', max_length=128, null=True)
    col_13 = models.CharField('属性13', max_length=128, null=True)
    col_14 = models.CharField('属性14', max_length=128, null=True)
    col_15 = models.CharField('属性15', max_length=128, null=True)
    col_16 = models.CharField('属性16', max_length=128, null=True)
    col_17 = models.CharField('属性17', max_length=128, null=True)
    col_18 = models.CharField('属性18', max_length=128, null=True)
    col_19 = models.CharField('属性19', max_length=128, null=True)
    col_20 = models.CharField('属性20', max_length=128, null=True)
    # todo 作用不知
    def __str__(self):
        return f'通用数据信息表'

    class Meta:
        db_table = "generalmaskinfo"

# todo 通用表-填补后
class GeneralFilledInfo(models.Model):
    table_name = models.CharField('表名', max_length=128, null=True, blank=True)
    col_1 = models.CharField('属性1', max_length=128, null=True, blank=True)
    col_2 = models.CharField('属性2', max_length=128, null=True, blank=True)
    col_3 = models.CharField('属性3', max_length=128, null=True, blank=True)
    col_4 = models.CharField('属性4', max_length=128, null=True, blank=True)
    col_5 = models.CharField('属性5', max_length=128, null=True, blank=True)
    col_6 = models.CharField('属性6', max_length=128, null=True, blank=True)
    col_7 = models.CharField('属性7', max_length=128, null=True, blank=True)
    col_8 = models.CharField('属性8', max_length=128, null=True, blank=True)
    col_9 = models.CharField('属性9', max_length=128, null=True, blank=True)
    col_10 = models.CharField('属性10', max_length=128, null=True, blank=True)
    col_11 = models.CharField('属性11', max_length=128, null=True, blank=True)
    col_12 = models.CharField('属性12', max_length=128, null=True, blank=True)
    col_13 = models.CharField('属性13', max_length=128, null=True, blank=True)
    col_14 = models.CharField('属性14', max_length=128, null=True, blank=True)
    col_15 = models.CharField('属性15', max_length=128, null=True, blank=True)
    col_16 = models.CharField('属性16', max_length=128, null=True, blank=True)
    col_17 = models.CharField('属性17', max_length=128, null=True, blank=True)
    col_18 = models.CharField('属性18', max_length=128, null=True, blank=True)
    col_19 = models.CharField('属性19', max_length=128, null=True, blank=True)
    col_20 = models.CharField('属性20', max_length=128, null=True, blank=True)
    # todo 作用不知
    def __str__(self):
        return f'通用数据填补表'

    class Meta:
        db_table = "generalfilledinfo"

    # def get_absolute_url(self):
    #     return reverse('upload_bank')