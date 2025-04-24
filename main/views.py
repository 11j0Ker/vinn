import hashlib
import json
import os
import random
import tempfile
import time
import pandas as pd
from collections import Counter
from datetime import datetime, timedelta
from pyecharts.charts import Pie, Bar, Line, Scatter3D, Radar
from pyecharts import options as opts
from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.http import HTTPRequest
from collections import defaultdict

from django.contrib.auth import logout
from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import render, redirect
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.contrib import messages

from main.DL import load_model, load_model_duofenlei2, load_model_duofenlei, load_model2
from main.DL.test_model_duofenlei2 import apply_model_on_test_file_muti
from main.DL.test_model_new2 import apply_model_on_test_file_single
from main.forms import CustomCaptchaForm
from main.models import User, Task, IPAddressRule, TuningModels
from main.firewall import FirewallManager
from main.utils import generate_task_id, http_attack
from dl_ids.settings import STATICFILES_DIRS


def login(request):
    captcha_form = CustomCaptchaForm()
    context = {
        "captcha_form": captcha_form,
    }
    return render(request, 'login.html', context)


def do_login(request):
    captcha_form = CustomCaptchaForm()
    try:
        user = User.objects.get(username=request.POST['username'])
        print(user.toDict())
        if user.status == 1:
            import hashlib
            md5 = hashlib.md5()
            s = request.POST['pass'] + user.password_salt
            md5.update(s.encode('utf-8'))
            # 新增验证码
            captcha_form = CustomCaptchaForm(request.POST)
            if captcha_form.is_valid():
                # print(captcha_form.cleaned_data)
                if user.password_hash == md5.hexdigest():
                    request.session['is_login'] = True
                    request.session['login_user'] = user.toDict()

                    # print("用户登录")
                    # print(request.session['login_user'])
                    return redirect('/index')
                else:
                    context = {
                        "captcha_form": captcha_form,
                        "info": '账号或密码错误'
                    }
                    messages.error(request, '账号或密码错误')
                    return render(request, 'login.html', context)
            else:
                context = {
                    "captcha_form": captcha_form,
                    "error": '验证码错误'
                }
                messages.error(request, '验证码错误')
                return render(request, 'login.html', context)
        if user.status == 6:
            import hashlib
            md5 = hashlib.md5()
            s = request.POST['pass'] + user.password_salt
            md5.update(s.encode('utf-8'))
            captcha_form = CustomCaptchaForm(request.POST)
            if captcha_form.is_valid():
                print("验证码正确")
                if user.password_hash == md5.hexdigest():
                    request.session['is_login'] = True
                    request.session['adminuser'] = user.toDict()
                    return redirect(reverse("myadmin_user_index", args=(1,)))
                else:
                    context = {
                        "captcha_form": captcha_form,
                        "info": '密码错误'
                    }
                    messages.error(request, '账号或密码错误')
                    return render(request, 'login.html', context)
        else:
            context = {
                "captcha_form": captcha_form,
                "info": '账号无权限'
            }
            messages.error(request, '账号无权限')
    except Exception as e:
        print("报错为", e)
        context = {
            "captcha_form": captcha_form,
            "info": '账号或密码错误'
        }
        messages.error(request, '账号或密码错误')
    context = {
        "captcha_form": captcha_form,
        "info": '账号或密码错误'
    }
    messages.error(request, '账号或密码错误')
    return render(request, 'login.html', context)


def logout_view(request):
    logout(request)  # 清除当前用户所有session
    print("用户session", request.session.get('is_login'))
    return HttpResponseRedirect('login')


def register(request):
    return render(request, 'register.html')


def do_register(request):
    captcha_form = CustomCaptchaForm()
    username = request.POST['username']

    # 检查用户名是否已存在
    if User.objects.filter(username=username).exists():
        # 用户名已存在，返回错误信息或重定向到适当的页面
        error_message = "该用户名已被注册，请选择其他用户名。"
        context = {
            "captcha_form": captcha_form,
            "error": error_message,
        }
        return render(request, 'register.html', context)

    import hashlib, random
    md5 = hashlib.md5()
    n = random.randint(100000, 999999)
    s = request.POST['pass'] + str(n)
    md5.update(s.encode('utf-8'))

    ob = User()
    ob.username = username
    ob.password_hash = md5.hexdigest()
    ob.password_salt = n
    ob.status = 1
    ob.create_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ob.update_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ob.save()

    context = {
        "captcha_form": captcha_form,
    }
    return render(request, 'login.html', context)


def forgot_pd(request):
    captcha_form = CustomCaptchaForm()

    if request.method == 'POST':
        username = request.POST['username']

        try:
            user = User.objects.get(username=username)

            if user.status != 1:
                context = {
                    "captcha_form": captcha_form,
                    "msg": '只有普通用户能够修改密码'
                }
                return render(request, 'forgot_pd.html', context)

            captcha_form = CustomCaptchaForm(request.POST)

            if captcha_form.is_valid():
                if not User.objects.filter(username=username).exists():
                    # 用户名不存在，返回错误信息或重定向到适当的页面
                    error_message = "该用户名未注册"
                    context = {
                        "captcha_form": captcha_form,
                        "msg": error_message,
                    }
                    return render(request, 'forgot_pd.html', context)
                else:
                    # 修改密码
                    md5 = hashlib.md5()
                    n = random.randint(100000, 999999)
                    s = request.POST['pass'] + str(n)
                    md5.update(s.encode('utf-8'))

                    # 更新用户密码信息
                    user.password_hash = md5.hexdigest()
                    user.password_salt = n
                    user.update_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    user.save()

                    context = {
                        "captcha_form": captcha_form,
                        "msg": '密码修改成功'
                    }
                    return render(request, 'forgot_pd.html', context)
            else:
                context = {
                    "captcha_form": captcha_form,
                    "msg": '验证码错误'
                }
                return render(request, 'forgot_pd.html', context)

        except User.DoesNotExist:
            # 用户不存在，返回错误信息或重定向到适当的页面
            error_message = "该用户名未注册"
            context = {
                "captcha_form": captcha_form,
                "msg": error_message,
            }
            return render(request, 'forgot_pd.html', context)

        except Exception as e:
            print(e)
            context = {
                "captcha_form": captcha_form,
                "msg": '内部错误'
            }
            return render(request, 'forgot_pd.html', context)

    context = {
        "captcha_form": captcha_form,
    }
    return render(request, 'forgot_pd.html', context)


def index(request, *args, **kwargs):
    # 返回页面时，需要判断用户是否已经开始扫描，若开始扫描则应该持续显示该页面
    user_id = request.session['login_user'].get('id')
    user = User.objects.get(id=user_id)

    return render(request, 'index.html')


@require_http_methods(["POST"])
def do_index(request):
    if request.method == 'POST':
        # 从 request 中获取上传的文件和数据
        file = request.FILES.get('file') if request.FILES else None
        model = request.POST.get('model')

        # 检查是否收到了文件
        if not file:
            return JsonResponse({'status': 'error', 'message': '未收到文件'}, status=400)
        # 创建任务ID
        task_id = generate_task_id()
        request.session['task_id'] = task_id
        # 创建临时文件    
        temp_file = tempfile.NamedTemporaryFile(dir=os.path.join(STATICFILES_DIRS[0], 'tmp'), delete=False)
        try:
            # 将上传的文件保存到临时文件
            for chunk in file.chunks():
                temp_file.write(chunk)
            temp_file.close()  # 关闭文件，以便可以被其他操作打开
            # 将扫描数据存入数据库
            user_id = request.session['login_user'].get('id')
            user = User.objects.get(id=user_id)
            # 获取临时文件的路径
            temp_file_path = temp_file.name
            print("保存路径为", temp_file_path)
            record = Task.objects.create(
                task_id=task_id,
                user=user,
                temp_result_file_path=temp_file_path,
                start_time=timezone.localtime(timezone.now()),
                end_time=timezone.localtime(timezone.now()),
            )
            record.save()

            res = {
                'status': 'success',
                'message': 'File and model data processed',
                'model_used': model,
                'path': temp_file_path,
            }
            return JsonResponse(res, status=200)
        except Exception as e:
            os.unlink(temp_file_path)  # 删除临时文件
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    else:
        return JsonResponse({'status': 'error', 'message': '方法不信任'}, status=405)


def predict_exec(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            model = data.get('model')
            path = data.get('path')
            if not path:
                return JsonResponse({'status': 'error', 'message': 'Path is Not Found'}, status=404)
            if model == "model1":
                # 处理数据处理
                accuracy, normal_count, abnormal_count = apply_model_on_test_file_single(path)  # 数据测试
                print(f"准确率: {accuracy * 100:.2f}%")  # 输出准确率

                # 更新任务状态
                task_id = request.session.get('task_id')
                if task_id:
                    Task.objects.filter(task_id=task_id).update(
                        status='completed',
                        exec_time=0 if 'exec_time' in locals() else 0,
                        end_time=timezone.localtime(timezone.now()),
                    )

                res = {
                    'status': 'success',
                    'message': 'Completed the calculation of test set accuracy.',
                    'overall_accuracy': f"{accuracy * 100:.2f}",  # 转换为浮点数后再乘以100
                    'normal_count': normal_count,
                    'abnormal_count': abnormal_count
                }
                return JsonResponse(res, status=200)
            elif model == "model2":
                # 处理数据处理
                accuracy, label_counts = apply_model_on_test_file_muti(path)  # 数据测试
                attack_types = list(label_counts.keys())  # 获取所有的攻击类型（键）
                attack_count_val = list(label_counts.values())  # 获取所有的攻击计数值（值）
                print(f"准确率: {accuracy * 100:.2f}%")  # 输出准确率

                # 更新任务状态
                task_id = request.session.get('task_id')
                if task_id:
                    Task.objects.filter(task_id=task_id).update(
                        status='completed',
                        exec_time=0 if 'exec_time' in locals() else 0,
                        end_time=timezone.localtime(timezone.now()),
                    )
                label_counts_array = [[key, value] for key, value in label_counts.items() if value != 0]

                res = {
                    'status': 'success',
                    'message': 'Completed the calculation of test set accuracy.',
                    'overall_accuracy': f"{accuracy * 100:.2f}",  # 转换为浮点数后再乘以100
                    'attack_types': attack_types,
                    'attack_count_val': label_counts_array
                }
                return JsonResponse(res, status=200)
            else:
                return JsonResponse(status=500)

        except Exception as e:
            print(f"预测过程中出错: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': f'处理失败: {str(e)}'
            }, status=500)


def screen(request):
    # 展示图像
    def bar_echarts(tuning_data, user_tuning_data_duofenlei):
        from pyecharts.charts import Bar
        # 创建 Bar 实例
        bar = Bar(init_opts=opts.InitOpts(width="100%", height="400px"))

        accuracy_data = [format(tuning_data[0], '.2f'), format(user_tuning_data_duofenlei[0], '.2f')]  # 准确率数据
        loss_data = [format(tuning_data[1], '.2f'), format(user_tuning_data_duofenlei[1], '.2f')]  # 精确率数据
        test_accuracy_data = [format(tuning_data[2], '.2f'), format(user_tuning_data_duofenlei[2], '.2f')]  # 召回率数据

        # 添加X轴数据
        x_axis_data = ['二分类模型', '多分类模型']
        bar.add_xaxis(x_axis_data)

        # 添加 Y 轴数据，这里可以添加多个系列，每个系列代表一个堆叠层
        bar.add_yaxis("验证集准确率", accuracy_data)
        bar.add_yaxis("损失值", loss_data)
        bar.add_yaxis("测试集准确率", test_accuracy_data)

        # 设置全局配置项，包括图注
        title_opts = opts.TitleOpts(title="模型性能指标堆叠柱状图")
        legend_opts = opts.LegendOpts(pos_top="8%")  # 设置图例位置在顶部
        bar.set_global_opts(title_opts, legend_opts)
        return bar.render_embed()

    def radar_echarts(tuning_data, user_tuning_data_duofenlei):
        """
        雷达图，显示模型总览
        """
        model = [[format(tuning_data[0], '.2f'), format(tuning_data[1], '.2f'), format(tuning_data[2], '.2f')]]
        model1 = [[format(user_tuning_data_duofenlei[0], '.2f'), format(user_tuning_data_duofenlei[1], '.2f'),
                   format(user_tuning_data_duofenlei[2], '.2f')]]
        # model = [[format(tuning_data[0], '.2f'), format(tuning_data[1], '.2f'), format(tuning_data[0], '.2f')]]
        # model1 = [[format(user_tuning_data_duofenlei[0], '.2f'), format(user_tuning_data_duofenlei[1], '.2f'),
        #            format(tuning_data[0], '.2f')]]
        Radar_echarts = (
            Radar(init_opts=opts.InitOpts(width="100%", height="400px"))
            .add_schema(
                schema=[
                    opts.RadarIndicatorItem(name="验证集准确率", max_=100),
                    opts.RadarIndicatorItem(name="损失值", max_=100),
                    opts.RadarIndicatorItem(name="测试集准确率", max_=100),
                ],
                splitarea_opt=opts.SplitAreaOpts(
                    is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
                textstyle_opts=opts.TextStyleOpts(color="#000"),
            )
            .add(
                series_name="二分类模型",
                data=model,
                linestyle_opts=opts.LineStyleOpts(color="#CD0000"),
            )
            .add(
                series_name="多分类模型",
                data=model1,
                linestyle_opts=opts.LineStyleOpts(color="#5CACEE"),
            )
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
                title_opts=opts.TitleOpts(title="模型综合雷达图"), legend_opts=opts.LegendOpts(pos_top="0%"),
            )
        )

        return Radar_echarts.render_embed()

    def scatter_3d_echarts(scatter_3d_data):
        lr = scatter_3d_data[0]
        num_epochs = scatter_3d_data[1]
        accuracy = scatter_3d_data[2]
        loss = scatter_3d_data[3]
        data_accuracy = [[n, d, a] for n, d, a in zip(lr, num_epochs, accuracy)]
        data_loss = [[n, d, e] for n, d, e in zip(lr, num_epochs, loss)]

        # 创建 3D 散点图
        scatter_3d = (
            Scatter3D(init_opts=opts.InitOpts(width="100%", height="600px"))
            .add(
                series_name="推理集准确率",
                data=data_accuracy,
                xaxis3d_opts=opts.Axis3DOpts(name='学习率'),
                yaxis3d_opts=opts.Axis3DOpts(name='训练轮数'),
                zaxis3d_opts=opts.Axis3DOpts(name='推理集准确率'),
                itemstyle_opts=opts.ItemStyleOpts(color='blue'),
            )
            .add(
                series_name="损失值",
                data=data_loss,
                xaxis3d_opts=opts.Axis3DOpts(name='学习率'),
                yaxis3d_opts=opts.Axis3DOpts(name='训练轮数'),
                zaxis3d_opts=opts.Axis3DOpts(name='损失值'),
                itemstyle_opts=opts.ItemStyleOpts(color='red'),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="参数调优关系图(最近七天)"),
                legend_opts=opts.LegendOpts(is_show=True, pos_top="8%"),
            )
        )
        return scatter_3d.render_embed()

    # 用户自训练的模型数据
    user_id = request.session['login_user'].get('id')
    user = User.objects.get(id=user_id)
    try:
        # 获取结果
        user_last_model = TuningModels.objects.filter(user=user, tuning_model='LSTM').latest('end_time')
        user_tuning_model = user_last_model.tuning_model
        accuracy = user_last_model.accuracy
        loss = user_last_model.loss
        test_accuracy = user_last_model.test_accuracy
        # 获取结果
        user_last_model2 = TuningModels.objects.filter(user=user, tuning_model='LSTM multiple').latest('end_time')
        user_tuning_model2 = user_last_model2.tuning_model
        accuracy2 = user_last_model2.accuracy
        loss2 = user_last_model2.loss
        test_accuracy2 = user_last_model2.test_accuracy
    except TuningModels.DoesNotExist:
        user_tuning_model, accuracy, precision, loss, test_accuracy = '', 0, 0, 0, 0
        user_last_model2, accuracy2, precision2, loss2, test_accuracy2 = '', 0, 0, 0, 0
    user_tuning_data = [accuracy, loss, test_accuracy, user_tuning_model]
    user_tuning_data_duofenlei = [accuracy2, loss2, test_accuracy2, user_tuning_model2]

    now = timezone.now()

    # 计算七天前的时间
    seven_days_ago = now - timedelta(days=7)
    # 查询最近七天的数据
    recent_tuning_models = TuningModels.objects.filter(user=user, start_time__gte=seven_days_ago).order_by('start_time')
    lr_list, num_epochs_list, accuracy_list, loss_list = [], [], [], []
    for i in range(len(recent_tuning_models)):
        recent_tuning_lr = recent_tuning_models[i].lr  # 学习率
        recent_tuning_num_epochs = recent_tuning_models[i].num_epochs  # 训练轮数
        recent_tuning_accuracy = recent_tuning_models[i].accuracy  # 推理集正确率
        recent_tuning_loss = recent_tuning_models[i].loss  # 推理集损失值

        lr_list.append(recent_tuning_lr)
        num_epochs_list.append(recent_tuning_num_epochs)
        accuracy_list.append(recent_tuning_accuracy)
        loss_list.append(recent_tuning_loss)
    scatter_3d_data = [lr_list, num_epochs_list, accuracy_list, loss_list]
    print("最近七天的数据为:", scatter_3d_data)
    data = {
        'status': 'success',
        'bar_chart': bar_echarts(user_tuning_data, user_tuning_data_duofenlei),
        'radar_chart': radar_echarts(user_tuning_data, user_tuning_data_duofenlei),
        'bar_3d': scatter_3d_echarts(scatter_3d_data)

    }
    return render(request, 'screen.html', data)


# def model_info(request):
#     data = {
#         'status': 'success',
#     }
#     return render(request, 'model_info.html', data)


def dataset_result(request):
    # 定义列名
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
        'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
        'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level'
    ]

    # 读取NSL-KDD测试集数据
    df = pd.read_csv(os.path.join(STATICFILES_DIRS[0], 'data_set', 'KDDTest', 'KDDTest+.txt'),
                     header=None, names=columns)

    # 将攻击类型映射为主要类别
    attack_mapping = {
        'normal': 'normal',
        'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos',
        'smurf': 'dos', 'teardrop': 'dos',
        'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe', 'satan': 'probe',
        'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l', 'multihop': 'r2l',
        'phf': 'r2l', 'spy': 'r2l', 'warezclient': 'r2l', 'warezmaster': 'r2l',
        'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r', 'rootkit': 'u2r'
    }
    df['attack_category'] = df['attack'].map(lambda x: attack_mapping.get(x, 'unknown'))
    print(f"攻击类型分布为：{df['attack_category']}")

    # 1. 攻击类型分布饼图
    attack_dist = (
        Pie(init_opts=opts.InitOpts(width="100%", height="400px"))
        .add(
            "",
            [list(z) for z in df['attack_category'].value_counts().items()],
            radius=["40%", "75%"],
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="攻击类型分布"),
            legend_opts=opts.LegendOpts(orient="vertical", pos_top="15%", pos_left="2%")
        )
    )

    # 2. 协议类型统计柱状图
    protocol_stats = (
        Bar(init_opts=opts.InitOpts(width="100%", height="400px"))
        .add_xaxis(df['protocol_type'].unique().tolist())
        .add_yaxis("数量", df['protocol_type'].value_counts().tolist())
        .set_global_opts(
            title_opts=opts.TitleOpts(title="协议类型统计"),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),
            toolbox_opts=opts.ToolboxOpts()
        )
    )

    # 3. 服务类型TOP10统计
    service_top10 = (
        Bar(init_opts=opts.InitOpts(width="100%", height="400px"))
        .add_xaxis(df['service'].value_counts().head(10).index.tolist())
        .add_yaxis("数量", df['service'].value_counts().head(10).values.tolist())
        .set_global_opts(
            title_opts=opts.TitleOpts(title="服务类型TOP10统计"),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),
            toolbox_opts=opts.ToolboxOpts()
        )
    )

    # 4. 连接时长分布折线图
    duration_stats = (
        Line(init_opts=opts.InitOpts(width="100%", height="400px"))
        .add_xaxis(
            [str(x) for x in range(10)]
        )
        .add_yaxis(
            "连接数量",
            df['duration'].value_counts().sort_index().head(10).values.tolist(),
            is_smooth=True
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="连接时长分布(前10个时间点)"),
            toolbox_opts=opts.ToolboxOpts()
        )
    )

    data = {
        'status': 'success',
        'attack_dist': attack_dist.dump_options(),
        'protocol_stats': protocol_stats.dump_options(),
        'service_top10': service_top10.dump_options(),
        'duration_stats': duration_stats.dump_options(),
    }

    return render(request, 'dataset_result.html', data)


def model_tuning(request):
    data = {
        'status': 'success',
    }
    return render(request, 'model_tuning.html', data)


def do_tuning_model(request):
    user_id = request.session['login_user'].get('id')
    user = User.objects.get(id=user_id)
    # 查询最新的调优任务记录
    latest_tuning_task = TuningModels.objects.filter(user=user, tuning_model='LSTM').order_by('-created_at').first()
    if latest_tuning_task:
        context = {
            'lr': latest_tuning_task.lr,
            'wd': latest_tuning_task.wd,
            'batch_size': latest_tuning_task.batch_size,
            'num_epochs': latest_tuning_task.num_epochs,
            'accuracy': latest_tuning_task.accuracy,
            'alpha': latest_tuning_task.alpha,
            'temperature': latest_tuning_task.temperature,
            'loss': latest_tuning_task.loss,
            'test_accuracy': latest_tuning_task.test_accuracy,
        }
    else:
        context = {}
    return render(request, 'do_tuning.html', context)


def tuning_lstm(request):
    if request.method == 'POST':
        try:
            # 从请求体中获取 JSON 数据
            data = json.loads(request.body)

            # 获取参数
            lr = float(data.get('lr', 5e-6))
            wd = float(data.get('wd', 6e-6))
            batch_size = int(data.get('batch_size', 256))
            num_epochs = int(data.get('num_epochs', 20))
            alpha = float(data.get('alpha', 0.5))
            temperature = float(data.get('temperature', 2))
            overSamplingValue = int(data.get('overSampling'))

            # 调用 LSTM 训练函数
            accuracy, loss, test_accuracy = load_model2.main(num_epochs=num_epochs, lr=lr, wd=wd, batch_size=batch_size,
                                                             alpha=alpha, temperature=temperature,
                                                             overSamplingValue=overSamplingValue)
            print("得到的就结果为：", accuracy, loss, test_accuracy)
            # 创建并保存 TuningModels 实例
            user_id = request.session['login_user'].get('id')
            user = User.objects.get(id=user_id)
            tuning_task = TuningModels.objects.create(
                tuning_id=generate_task_id(),
                user=user,
                tuning_model='LSTM',
                start_time=timezone.localtime(timezone.now()),
                end_time=timezone.localtime(timezone.now()),
                lr=lr,
                wd=wd,
                batch_size=batch_size,
                num_epochs=num_epochs,
                alpha=alpha,
                temperature=temperature,
                accuracy=f"{accuracy * 100:.2f}",
                loss=f"{loss:.2f}",
                test_accuracy=f"{test_accuracy:.2f}"
            )
            tuning_task.save()
            # 返回训练结果
            return JsonResponse({
                'status': 'success',
                'accuracy': f"{accuracy * 100:.2f}",
                'loss': f"{loss:.2f}",
                'test_accuracy': f"{test_accuracy:.2f}",
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    else:
        return JsonResponse({'status': 'error', 'message': 'Only POST requests are allowed.'}, status=405)


def reset_parameter_lstm(request):
    # 重置参数为默认值
    default_params = {
        'lr': 5e-5,
        'wd': 6e-6,
        'batch_size': 256,
        'num_epochs': 20,
        'alpha': 0.5,
        'temperature': 2
    }
    return render(request, 'do_tuning.html', context=default_params)


def reset_parameter_lstm2(request):
    # 重置参数为默认值
    default_params = {
        'lr': 8e-3,
        'wd': 6e-6,
        'batch_size': 256,
        'num_epochs': 10,
        'alpha': 0.5,
        'temperature': 2
    }
    return render(request, 'do_tuning_duofenlei.html', context=default_params)


def upload_pcapng(request):
    from decimal import Decimal
    from datetime import datetime
    import json
    start_time = time.time()

    def res_unique_rm_city(data, is_attack_data=False):
        """
        处理数据去重
        :param data: 要处理的数据列表
        :param is_attack_data: 是否是攻击类型数据（结构更简单）
        """
        unique_values = set()
        unique_data = []

        if is_attack_data:
            # 处理攻击类型数据 [source_ip, attack_type]
            for item in data:
                if item[1] not in unique_values:
                    unique_values.add(item[1])
                    unique_data.append(item)

            counter = Counter([item[1] for item in data])
            for item in unique_data:
                item.append(counter[item[1]])  # 添加计数
            return unique_data
        else:
            # 处理详细数据包信息 [source_ip, dest_ip, url, attack_type, packet_info]
            for item in data:
                unique_key = tuple(item[:4])
                if unique_key not in unique_values:
                    unique_values.add(unique_key)
                    unique_data.append(item)

            counter = Counter(tuple(item[:4]) for item in data)

            for item in unique_data:
                unique_key = tuple(item[:4])
                count = counter[unique_key]
                packet_info = item[4] if len(item) > 4 else {}

                new_item = [
                    item[0],  # source_ip
                    item[1],  # destination_ip
                    item[2],  # url
                    item[3],  # attack_type
                    count,  # 计数
                    packet_info  # 详细信息字典
                ]
                item[:] = new_item

            return unique_data

    if request.method == 'POST' and request.FILES['file']:
        file = request.FILES['file']
        file_name = 'file.pcap'  # 指定文件名
        # 为了保证始终只有一个文件，则将目录中的文件先删除在保存
        file_path = os.path.join(default_storage.location, file_name)
        if default_storage.exists(file_path):
            default_storage.delete(file_path)

        # 保存新的文件
        file_path = default_storage.save(file_name, file)
        file_path = os.path.join(default_storage.location, file_path)
        packets = rdpcap(file_path)  # 读取所有数据包

        # 统计所有数据包信息
        packet_stats = {
            'total_packets': len(packets),
            'protocols': defaultdict(int),
            'http_packets': 0,
            'tcp_packets': 0,
            'udp_packets': 0,
            'other_packets': 0,
            'total_bytes': 0,
            'avg_packet_size': 0
        }

        # 分析每个数据包
        for packet in packets:
            packet_stats['total_bytes'] += len(packet)

            if packet.haslayer(TCP):
                packet_stats['tcp_packets'] += 1
                if packet.haslayer(HTTPRequest):
                    packet_stats['http_packets'] += 1
            elif packet.haslayer(UDP):
                packet_stats['udp_packets'] += 1
            else:
                packet_stats['other_packets'] += 1

            # 获取协议名称
            if packet.haslayer(IP):
                proto_name = packet[IP].get_field('proto').i2s.get(packet[IP].proto, str(packet[IP].proto))
                packet_stats['protocols'][proto_name] += 1

        # 计算平均包大小
        packet_stats['avg_packet_size'] = round(packet_stats['total_bytes'] / packet_stats['total_packets'], 2)

        # 将协议统计转换为图表数据
        protocol_data = [{"name": proto, "value": count}
                         for proto, count in packet_stats['protocols'].items()]
        protocol_names = json.dumps([item['name'] for item in protocol_data])
        protocol_values = json.dumps([item['value'] for item in protocol_data])

        data = []
        attack_unique_data = []
        total_packets = len(packets)  # 总数据包数
        attack_packets = 0  # 攻击数据包数

        for packet in packets:
            ip_layer = packet.getlayer(IP)
            tcp_layer = packet.getlayer(TCP)
            http_layer = packet.getlayer(HTTPRequest)

            if ip_layer is not None:
                # 转换时间戳
                try:
                    timestamp = float(packet.time)
                    formatted_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')
                except (TypeError, ValueError):
                    formatted_time = "Unknown Time"

                # 初始化基本信息
                packet_info = {
                    'source_ip': ip_layer.src,
                    'destination_ip': ip_layer.dst,
                    'timestamp': formatted_time,
                    'length': len(packet),
                    'source_mac': packet.src,
                    'dest_mac': packet.dst,
                    'protocol': packet.proto,
                    'packet_id': packet.id,
                }

                # 添加 TCP 相关信息（如果存在）
                if tcp_layer is not None:
                    packet_info.update({
                        'source_port': tcp_layer.sport,
                        'destination_port': tcp_layer.dport,
                    })
                else:
                    packet_info.update({
                        'source_port': 'N/A',
                        'destination_port': 'N/A',
                    })

                # 添加 HTTP 相关信息（如果存在）
                if http_layer is not None:
                    packet_info.update({
                        'method': http_layer.Method.decode(),
                        'url': http_layer.Host.decode() + http_layer.Path.decode(),
                        'user_agent': http_layer.fields.get('User-Agent', b'').decode('utf-8', 'ignore'),
                        'referer': http_layer.fields.get('Referer', b'').decode('utf-8', 'ignore'),
                    })
                else:
                    packet_info.update({
                        'method': 'N/A',
                        'url': 'N/A',
                        'user_agent': 'N/A',
                        'referer': 'N/A',
                    })

                # 匹配url是否为恶意流量（仅对HTTP包进行检查）
                if http_layer is not None:
                    is_attack = http_attack(packet_info['url'])
                    if is_attack[1] is not None:
                        attack_packets += 1
                        attack_unique_data.append([packet_info['source_ip'], is_attack[0][0]])
                        packet_info['attack'] = is_attack[0][0]
                        data.append([
                            packet_info['source_ip'],
                            packet_info['destination_ip'],
                            packet_info['url'],
                            packet_info['attack'],
                            packet_info
                        ])
            else:
                continue

        table_val = res_unique_rm_city(data)  # 处理详细数据

        # 计算正常流量数
        normal_packets = total_packets - attack_packets

        # 处理攻击类型数据
        if len(attack_unique_data) != 0:
            datas = res_unique_rm_city(attack_unique_data, is_attack_data=True)
            attack_type = []
            for vul in datas:
                i = {"value": vul[2], "name": vul[1]}
                attack_type.append(i)
            # 将数据转换为JSON字符串
            attack_names = json.dumps([item['name'] for item in attack_type])
            attack_values = json.dumps([item['value'] for item in attack_type])
        else:
            attack_names = '[]'
            attack_values = '[]'
            attack_type = []

        # 计算总流量数据
        total_data = {
            'total_packets': total_packets,
            'attack_packets': attack_packets,
            'normal_packets': normal_packets,
            'http_packets': packet_stats['http_packets'],
            'tcp_packets': packet_stats['tcp_packets'],
            'udp_packets': packet_stats['udp_packets'],
            'other_packets': packet_stats['other_packets'],
            'total_bytes': packet_stats['total_bytes'],
            'avg_packet_size': packet_stats['avg_packet_size']
        }

        # 计算威胁检出率
        threat_rate = round((attack_packets / total_packets * 100), 2) if total_packets > 0 else 0

        end_time = time.time()
        run_time = round((end_time - start_time) / 60, 2)
        is_upload = 'is_upload'
        return render(request, 'upload_pcapng.html',
                      {'data': table_val,
                       'attack_names': attack_names,
                       'attack_values': attack_values,
                       'protocol_names': protocol_names,
                       'protocol_values': protocol_values,
                       'attack_data': {'attack_type': attack_type},
                       'total_data': total_data,
                       'is_upload': is_upload,
                       'run_time': run_time,
                       'threat_rate': threat_rate})
    is_upload = 'is_upload'
    return render(request, 'upload_pcapng.html', {'is_upload': is_upload})


def ip_rule_list(request):
    """显示IP规则列表"""
    whitelist = IPAddressRule.objects.filter(rule_type='white')
    blacklist = IPAddressRule.objects.filter(rule_type='black')

    context = {
        'whitelist': whitelist,
        'blacklist': blacklist
    }
    return render(request, 'myadmin/ip_rules.html', context)


def add_ip_rule(request):
    """添加IP规则"""
    if request.method == 'POST':
        ip_address = request.POST.get('ip_address')
        rule_type = request.POST.get('rule_type')
        description = request.POST.get('description', '')

        try:
            # 检查IP是否已存在
            if IPAddressRule.objects.filter(ip_address=ip_address).exists():
                messages.error(request, f'IP地址 {ip_address} 已存在于规则列表中')
                return redirect('ip_rule_list')

            # 添加防火墙规则
            if FirewallManager.add_rule(ip_address, rule_type):
                # 保存到数据库
                IPAddressRule.objects.create(
                    ip_address=ip_address,
                    rule_type=rule_type,
                    description=description
                )
                messages.success(request, f'成功添加{rule_type}规则：{ip_address}')
            else:
                messages.error(request, f'添加防火墙规则失败：{ip_address}')

        except ValueError as e:
            messages.error(request, str(e))
        except Exception as e:
            messages.error(request, f'添加规则时发生错误：{str(e)}')

    return redirect('ip_rule_list')


def delete_ip_rule(request, rule_id):
    """删除IP规则"""
    try:
        rule = IPAddressRule.objects.get(id=rule_id)

        # 删除防火墙规则
        if FirewallManager.remove_rule(rule.ip_address, rule.rule_type):
            # 从数据库中删除
            rule.delete()
            messages.success(request, f'成功删除规则：{rule.ip_address}')
        else:
            messages.error(request, f'删除防火墙规则失败：{rule.ip_address}')

    except IPAddressRule.DoesNotExist:
        messages.error(request, '规则不存在')
    except Exception as e:
        messages.error(request, f'删除规则时发生错误：{str(e)}')

    return redirect('ip_rule_list')


def do_tuning_model_duofenlei(request):
    user_id = request.session['login_user'].get('id')
    user = User.objects.get(id=user_id)
    # 查询最新的调优任务记录
    latest_tuning_task = TuningModels.objects.filter(user=user, tuning_model='LSTM multiple').order_by(
        '-created_at').first()
    if latest_tuning_task:
        context = {
            'lr': latest_tuning_task.lr,
            'wd': latest_tuning_task.wd,
            'batch_size': latest_tuning_task.batch_size,
            'num_epochs': latest_tuning_task.num_epochs,
            'accuracy': latest_tuning_task.accuracy,
            'loss': latest_tuning_task.loss,
            'test_accuracy': latest_tuning_task.test_accuracy,
        }
    else:
        context = {}
    return render(request, 'do_tuning_duofenlei.html', context)


def tuning_lstm_duofenlei(request):
    if request.method == 'POST':
        try:
            # 从请求体中获取 JSON 数据
            data = json.loads(request.body)

            # 获取参数
            lr = float(data.get('lr', 8e-5))  # 设置默认值为 8e-5
            wd = float(data.get('wd', 6e-6))  # 设置默认值为 6e-6
            batch_size = int(data.get('batch_size', 256))  # 设置默认值为 256
            num_epochs = int(data.get('num_epochs', 20))  # 设置默认值为 20
            alpha = float(data.get('alpha', 0.5))  # 设置默认值为 0.5
            temperature = float(data.get('temperature', 2.0))  # 设置默认值为 2.0
            overSamplingValue = int(data.get('overSampling'))
            # 调用 LSTM 训练函数
            accuracy, loss, test_accuracy = load_model_duofenlei2.main(num_epochs=num_epochs, lr=lr, wd=wd,
                                                                       batch_size=batch_size,
                                                                       alpha=alpha, temperature=temperature,
                                                                       overSamplingValue=overSamplingValue)
            print("得到的就结果为：", accuracy, loss, test_accuracy)
            # 创建并保存 TuningModels 实例
            user_id = request.session['login_user'].get('id')
            user = User.objects.get(id=user_id)
            tuning_task = TuningModels.objects.create(
                tuning_id=generate_task_id(),
                user=user,
                tuning_model='LSTM multiple',
                start_time=timezone.localtime(timezone.now()),
                end_time=timezone.localtime(timezone.now()),
                lr=lr,
                wd=wd,
                batch_size=batch_size,
                num_epochs=num_epochs,
                alpha=alpha,
                temperature=temperature,
                accuracy=f"{accuracy * 100:.2f}",
                loss=f"{loss:.2f}",
                test_accuracy=f"{test_accuracy:.2f}"
            )
            tuning_task.save()
            # 返回训练结果
            return JsonResponse({
                'status': 'success',
                'accuracy': f"{accuracy * 100:.2f}",
                'loss': f"{loss:.2f}",
                'test_accuracy': f"{test_accuracy:.2f}",
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    else:
        return JsonResponse({'status': 'error', 'message': 'Only POST requests are allowed.'}, status=405)


def ids_model(request):
    return render(request, 'model_info.html')
