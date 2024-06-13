# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.forms.utils import ErrorList
from django.http import HttpResponse
from .forms import LoginForm, SignUpForm, PasswordChForm
from book.models import UserInfo

def login_view(request):
    form = LoginForm(request.POST or None)

    msg = None

    if request.method == "POST":

        if form.is_valid():
            username = form.cleaned_data.get("username")
            password = form.cleaned_data.get("password")
            # todo 用户
            user = authenticate(username=username, password=password)
            # todo 管理员
            user_info = UserInfo.objects.filter(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect("upload_bank")
            elif user is not None and user_info is not None:
                login(request, user)
                return redirect("user_info")
            else:    
                msg = '用户名或密码出错'
        else:
            msg = 'From文件出错'

    return render(request, "accounts/login.html", {"form": form, "msg" : msg})

# def register_user(request):
#
#     msg     = None
#     success = False
#
#     if request.method == "POST":
#         form = SignUpForm(request.POST)
#         if form.is_valid():
#             form.save()
#             username = form.cleaned_data.get("username")
#             raw_password = form.cleaned_data.get("password1")
#             user = authenticate(username=username, password=raw_password)
#
#             # msg     = '用户已创建 - 请登录 <a href="/login">login</a>.'
#             msg = '用户已创建 - 请登录'
#             success = True
#
#             # todo 注册结束的时候保存数据到 userinfo
#             UserInfo.objects.create(username=username,
#                                     password=raw_password,
#                                     gender="待完善",
#                                     addr="待完善",
#                                     email="待完善",
#                                     phone=0)
#
#             #return redirect("/login/")
#
#
#         else:
#             msg = 'Form is not valid'
#     else:
#         form = SignUpForm()
#
#     return render(request, "accounts/register.html", {"form": form, "msg" : msg, "success" : success })


def register_user(request):

    form = PasswordChForm(request.POST or None)
    msg = None
    success = False

    if request.method == "POST":

        if form.is_valid():
            username = form.cleaned_data.get("username")
            password = form.cleaned_data.get("password")
            password1 = form.cleaned_data.get("password1")
            password2 = form.cleaned_data.get("password2")
            success = True
            # todo 用户
            user = authenticate(username=username, password=password)
            user_info = UserInfo.objects.filter(username=username, password=password)
            if user or user_info:
                if password1 != password2:
                    msg = '两次输入新密码不一致'
                else:
                    msg = '密码修改成功'
                    UserInfo.objects.filter(username=username).update(password=password1)
                    u = User.objects.get(username=username)
                    u.set_password(password1)
                    u.save()

            else:
                msg = '用户不存在 或 原密码输入错误'
        else:
            msg = 'From文件出错'

    return render(request, "accounts/register.html", {"form": form, "msg" : msg, "success" : success })
