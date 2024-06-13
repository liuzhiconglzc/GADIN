
from django.contrib import admin
# todo 这里和视频里的不太一样
from django.urls import path, include  # add this
from .views import HomeView,BookListView,BookCreateView,BookDeleteView,BookDetailView,BookUpdateView
from .views import CategoryListView,CategoryCreateView,CategoryDeleteView
from .views import PublisherListView,PublisherCreateView,PublisherDeleteView,PublisherUpdateView
from .views import ActivityListView,ActivityDeleteView
from .views import MemberCreateView,MemberUpdateView,MemberDeleteView,MemberListView,MemberDetailView
from .views import ProfileDetailView,ProfileCreateView,ProfileUpdateView
from django.conf import settings
from django.conf.urls.static import static
from .views import BorrowRecordListView,BorrowRecordCreateView,BorrowRecordDeleteView,BorrowRecordDetailView,auto_member,auto_book,BorrowRecordClose
from .views import DataCenterView,download_data, UserInfoView, UserUpdateView, UserDeleteView, UserCreateView
from .views import ChartView,global_serach,EmployeeView,EmployeeDetailView,EmployeeUpdate,NoticeListView,NoticeUpdateView
from .views import uploadexcel, upload_bank,BankInfoView, BankUpdateView, BankDeleteView, BankInfoDelView
from .views import BankInfoNovView, bankinfo_nov, BankInfoConView, bankinfo_con, ModelBuildView
from .views import data_clear, output_save, FillDataView, data_fill, data_clear_fill, dataclear, fill_save
from .views import power_show, filled_save
# TODO hmf app url路由接口文件
# todo urls.py对应views.py ; urls的路径别名，可对应前端界面的href链接
# todo views文件方法，是前台界面回传的具体实现方法
urlpatterns = [

    # todo name是为了反向解析，可定位html中的元素，可在href中用这里的url的name
    # HomePage todo 首页改变
    # path("",HomeView.as_view(), name='home'),
    # Book
    # path('book-list',BookListView.as_view(),name="book_list"),
    # path('book-create',BookCreateView.as_view(),name="book_create"),
    # path('book-update/<int:pk>/',BookUpdateView.as_view(),name="book_update"),
    # path('book-delete/<int:pk>/',BookDeleteView.as_view(),name="book_delete"),
    # path('book-detail/<int:pk>/',BookDetailView.as_view(),name="book_detail"),

    # todo 用户首页 用戶信息管理 UserInfoView
    path("",UserInfoView.as_view(), name='home'),
    # # todo 管理员首页
    # path("",UserInfoNewView.as_view(), name='home_admin'),

    # todo 用户信息管理
    path('user-info/', UserInfoView.as_view(), name="user_info"),
    path('user-update/<int:pk>/',UserUpdateView.as_view(),name="user_update"),
    path('user-delete/<int:pk>/',UserDeleteView.as_view(),name="user_delete"),
    path('user-create', UserCreateView.as_view(),name="user_create"),
    # todo upload_bank 两种方式二选一
    # path('upload_bank/', uploadexcel, name='upload_bank'),
    # todo 数据重编码   用本地方式
    path('bank-info/', BankInfoView.as_view(), name="upload_bank"),
    # todo 导入数据
    path('upload/', upload_bank, name='upload'),

    # todo 修改和删除数据
    path('bankinfo-update/<int:pk>/', BankUpdateView.as_view(), name="bank_update"),
    path('bank-delete/<int:pk>/',BankDeleteView.as_view(),name="bank_delete"),
    # todo 冗余属性删除
    path('bank-info-del/', BankInfoDelView.as_view(), name="del_attribute"),

    # todo 数据归一化
    path('bank-info-nov/', BankInfoNovView.as_view(), name="data_nov"),
    path('bank-nov/', bankinfo_nov, name="data_nov_submit"),

    # todo 计算缺失率
    path('bank-info-con/', BankInfoConView.as_view(), name="data_con"),
    path('bank-con/', bankinfo_con, name="data_con_submit"),

    # todo 模型构建
    path('model-build/', ModelBuildView.as_view(), name="model_building"),
    # todo 清空数据
    path('data-clear/', data_clear, name="data_clear"),
    # todo 保存模型--改成-- 导出数据的一般通用方法
    path('output-save/', output_save, name="output_excel"),

    # todo 填补数据
    path('fill-data/', FillDataView.as_view(), name="fill_data"),
    path('fill-data-submit/', data_fill, name="fill_data_submit"),
    path('fill-data-before/', data_clear_fill, name="data_clear_fill"),
    # todo data_fill_save 保存数据方法暂时不用
    path('fill-data-save/', fill_save, name="data_fill_save"),
    path('fill-data-after/', dataclear, name="data_clear_filled"),
    path('fill-data-output/', filled_save, name="filled_output_excel"),



    # todo 用户权限提示 power_show
    path('show/', power_show, name="show"),




    # path('user/profile-create/',ProfileCreateView.as_view(),name="profile_create"),

    # Category
    path('category-list',CategoryListView.as_view(),name="category_list"),
    path('category-create',CategoryCreateView.as_view(),name="category_create"),  
    path('category-delete/<int:pk>/',CategoryDeleteView.as_view(),name="category_delete"), 

    # Publisher
    path('publisher-list',PublisherListView.as_view(),name="publisher_list"),
    path('publisher-create',PublisherCreateView.as_view(),name="publisher_create"),  
    path('publisher-delete/<int:pk>/',PublisherDeleteView.as_view(),name="publisher_delete"), 
    path('publisher-update/<int:pk>/',PublisherUpdateView.as_view(),name="publisher_update"),

    # User Activity
    path('user-activity-list',ActivityListView.as_view(),name="user_activity_list"),
    path('user-activity-list/<int:pk>/',ActivityDeleteView.as_view(),name="user_activity_delete"),

    # Membership
    path('member-list',MemberListView.as_view(),name="member_list"),
    path('member-create',MemberCreateView.as_view(),name="member_create"),  
    path('member-delete/<int:pk>/',MemberDeleteView.as_view(),name="member_delete"), 
    path('member-update/<int:pk>/',MemberUpdateView.as_view(),name="member_update"),
    path('member-detail/<int:pk>/',MemberDetailView.as_view(),name="member_detail"),

    # UserProfile
    path('user/profile-create/',ProfileCreateView.as_view(),name="profile_create"),
    path('user/<int:pk>/profile/',ProfileDetailView.as_view(),name="profile_detail"),
    path('user/<int:pk>/profile-update/',ProfileUpdateView.as_view(),name="profile_update"),


    # BorrowRecords
    path('record-create/',BorrowRecordCreateView.as_view(),name="record_create"),
    # path('record-create/',record_create,name="record_create"),

    path('record-create-autocomplete-member-name/',auto_member,name="auto_member_name"),
    path('record-create-autocomplete-book-name/',auto_book,name="auto_book_name"),
    path('record-list/',BorrowRecordListView.as_view(),name="record_list"),
    path('record-detail/<int:pk>/',BorrowRecordDetailView.as_view(),name="record_detail"),
    path('record-delete/<int:pk>/',BorrowRecordDeleteView.as_view(),name="record_delete"),
    path('record-close/<int:pk>/',BorrowRecordClose.as_view(),name="record_close"),

    # Data center
    path('data-center/',DataCenterView.as_view(),name="data_center"),
    path('data-download/<str:model_name>/',download_data,name="data_download"),

    # Chart
    path('charts/',ChartView.as_view(),name="chart"),

    # Global Search
    path('global-search/',global_serach,name="global_search"),

    # Employee
    path('employees/',EmployeeView.as_view(),name="employees_list"),
    path('employees-detail/<int:pk>',EmployeeDetailView.as_view(),name="employees_detail"),
    path('employees-update/<int:pk>',EmployeeUpdate,name='employee_update'),

    # Notice
    path('notice-list/', NoticeListView.as_view(), name='notice_list'),
    path('notice-update/', NoticeUpdateView.as_view(), name='notice_update'),
]



