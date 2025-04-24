from django.urls import path
from . import user

from main import views

urlpatterns = [
    # 基础服务
    path('', views.login, name="myadmin_login"),
    path('login', views.login, name="myadmin_login2"),
    path('dologin', views.do_login, name="myadmin_do_login"),
    path('logout', views.logout_view, name="myadmin_logout"),
    path('register', views.register, name="myadmin_register"),
    path('doregister', views.do_register, name="myadmin_do_register"),
    path('forgot_pd', views.forgot_pd, name="forgot_pd"),
    # 管理员
    path('user/<int:pIndex>', user.index, name="myadmin_user_index"),
    path('user/insert', user.insert, name="myadmin_user_insert"),
    path('user/delete/<int:uid>', user.delete, name="myadmin_user_delete"),
    path('user/edit/<int:uid>', user.edit, name="myadmin_user_edit"),
    path('user/update/<int:uid>', user.update, name="myadmin_user_update"),

    # 模型展示
    path('index', views.index, name="ids_index"),
    path('do_index', views.do_index, name="do_index"),
    path('screen', views.screen, name="ids_screen"),

    # 模型测试集
    path('dataset_res', views.dataset_result, name="ids_dataset_result"),
    path('ids_model', views.ids_model, name="ids_model"),

    # 模型使用
    path('predict_exec', views.predict_exec, name="predict_exec"),

    # 模型调优
    path('model_tuning', views.model_tuning, name="ids_model_tuning"),
    path('do_tuning_model', views.do_tuning_model, name="do_tuning_model"),
    path('do_tuning_model_duofenlei', views.do_tuning_model_duofenlei, name="do_tuning_model_duofenlei"),
    path('tuning_lstm/', views.tuning_lstm, name='tuning_lstm'),
    path('tuning_lstm_duofenlei/', views.tuning_lstm_duofenlei, name='tuning_lstm_duofenlei'),
    path('reset_parameter_lstm/', views.reset_parameter_lstm, name='reset_parameter_lstm'),
    path('reset_parameter_lstm2/', views.reset_parameter_lstm2, name='reset_parameter_lstm2'),


    # 本地pcap包自测
    path('upload_pcapng', views.upload_pcapng, name='upload_pcapng'),

    # IP规则管理
    path('ip-rules/', views.ip_rule_list, name='ip_rule_list'),
    path('ip-rules/add/', views.add_ip_rule, name='add_ip_rule'),
    path('ip-rules/delete/<int:rule_id>/', views.delete_ip_rule, name='delete_ip_rule'),


]