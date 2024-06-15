from django import forms
from .models import Book,Publisher,Member,Profile,BorrowRecord, UserInfo, BankInfo
from django.contrib.admin.widgets import AutocompleteSelect
from django.contrib import admin
from django.urls import reverse
from flatpickr import DatePickerInput, TimePickerInput, DateTimePickerInput
from .models import YhzxInfo, GeneralInfo

class BookCreateEditForm(forms.ModelForm):
    class Meta:
        model = Book
        fields = ('author',
                  'title',
                  'description',
                  'quantity', 
                  'category',
                  'publisher',
                  'floor_number',
                  "bookshelf_number")


# todo 用户信息的更新，添加
class UserCreateEditForm(forms.ModelForm):
    class Meta:
        model = UserInfo
        fields = ('username',
                  'password',
                  'gender',
                  # 'admin',
                  'phone',
                  'email',
                  'addr')

# todo 数据信息的更新，添加
class BankCreateEditForm(forms.ModelForm):
    class Meta:
        model = BankInfo
        fields = (
                  'company_name',
                  'type',
                  'bbd_url',
                  'bbd_type',
                  'cash_central_bank_funds',
                  'sellable_assetset',
                  'risk_preparation',
                  'interest_payable',
                  'paid_in_capital',
                  'fixed_assets',
                  'total_assets',
                  'capital_reserves')


class PubCreateEditForm(forms.ModelForm):
    class Meta:
        model = Publisher
        fields = ('name',
                  'city',
                  'contact',
                  )
        # fields="__all__"

class MemberCreateEditForm(forms.ModelForm):
    class Meta:
        model = Member
        fields = ('name',
                  'gender',
                  'age',
                  'email',
                  'city', 
                  'phone_number',)


class ProfileForm(forms.ModelForm):

    
    class Meta:
        model = Profile
        fields = ( 'profile_pic',
                  'bio', 
                  'phone_number',
                  'email')


class BorrowRecordCreateForm(forms.ModelForm):

    borrower = forms.CharField(label='Borrrower', 
                    widget=forms.TextInput(attrs={'placeholder': 'Search Member...'}))
    
    book = forms.CharField(help_text='type book name')

    class Meta:
        model = BorrowRecord
        fields=['borrower','book','quantity','start_day','end_day']
        # widgets = {
        #     'start_day': DatePickerInput().start_of('event datetime'),
        #     'end_day': DatePickerInput().end_of('event datetime'),
        # }
        widgets = {
            'start_day': DatePickerInput(options = {  "dateFormat": "Y-m-d", }),
            'end_day': DatePickerInput(options = {  "dateFormat": "Y-m-d", }),
        }
        # widgets = {'start_day': forms.DateTimeInput(attrs={'class': 'datepicker'}),
        #            'end_day': forms.DateTimeInput(attrs={'class': 'datepicker'})}


        # widgets = {
        #     'start_day': DateTimePickerInput(format='%Y-%m-%d'),
        #     'end_day': DateTimePickerInput(format='%Y-%m-%d'),
        # }


# from  django.forms.widgets import SelectDateWidget

# class BorrowRecordCreateForm(forms.ModelForm):

#     def __init__(self, *args, **kwargs):
#         super(BorrowRecordCreateForm, self).__init__(*args, **kwargs)
#         #Change date field's widget here
#         self.fields['start_day'].widget = SelectDateWidget()
#         self.fields['end_day'].widget = SelectDateWidget()

#     class Meta:
#         model = BorrowRecord
#         fields=['borrower','book','quantity','start_day','end_day']


#todo excel 批量上传功能
class UploadBankForm(forms.Form):
    uploadfile = forms.FileField()
    name = forms.CharField(max_length=50)


# todo 数据信息的更新，添加
class YhzxCreateEditForm(forms.ModelForm):
    class Meta:
        model = YhzxInfo
        fields = (
                  'company_name',
                  'capital_adequacy_ratio',
                  'provision_coverage',
                  'total_deposit',
                  'total_loan',
                  'non_interest_income',
                  'net_interest_margin')

# todo 数据信息的更新，添加
class GeneralCreateEditForm(forms.ModelForm):
    class Meta:
        model = GeneralInfo
        col_1 = forms.CharField(required=False)
        col_2 = forms.CharField(required=False)
        col_3 = forms.CharField(required=False)
        col_4 = forms.CharField(required=False)
        col_5 = forms.CharField(required=False)
        col_6 = forms.CharField(required=False)
        col_7 = forms.CharField(required=False)
        col_8 = forms.CharField(required=False)
        col_9 = forms.CharField(required=False)
        col_10 = forms.CharField(required=False)
        col_11 = forms.CharField(required=False)
        col_12 = forms.CharField(required=False)
        col_13 = forms.CharField(required=False)
        col_14 = forms.CharField(required=False)
        col_15 = forms.CharField(required=False)
        col_16 = forms.CharField(required=False)
        col_17 = forms.CharField(required=False)
        col_18 = forms.CharField(required=False)
        col_19 = forms.CharField(required=False)
        col_20 = forms.CharField(required=False)

        fields = (
                  'col_1',
                  'col_2',
                  'col_3',
                  'col_4',
                  'col_5',
                  'col_6',
                  'col_7',
                  'col_8',
                  'col_9',
                  'col_10',
                  'col_11',
                  'col_12',
                  'col_13',
                  'col_14',
                  'col_15',
                  'col_16',
                  'col_17',
                  'col_18',
                  'col_19',
                  'col_20'
        )