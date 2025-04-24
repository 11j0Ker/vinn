from django.db import models
from datetime import datetime

from django.utils.html import escape


class User(models.Model):
    username = models.CharField(max_length=50)
    nickname = models.CharField(max_length=50)
    password_hash = models.CharField(max_length=100)  # 密码
    password_salt = models.CharField(max_length=50)  # 密码干扰值
    status = models.IntegerField(default=1)  # 1正常 2禁用 6管理员 9删除
    create_at = models.DateTimeField(default=datetime.now)
    update_at = models.DateTimeField(default=datetime.now)
    is_authorize = models.BooleanField(default=False)
    is_change_file_type = models.BooleanField(default=False)  # 默认生成的文件类型为pdf

    def toDict(self):
        return {'id': self.id, 'username': self.username, 'nickname': self.nickname,
                'password_hash': self.password_hash, 'password_salt': self.password_salt, 'status': self.status,
                'create_at': self.create_at.strftime('%Y-%m-%d %H:%M:%S'),
                'update_at': self.update_at.strftime('%Y-%m-%d %H:%M:%S'),
                'is_authorize': self.is_authorize}

    class Meta:
        db_table = 'user'


class Task(models.Model):
    # 用于存储扫描任务
    task_id = models.UUIDField(unique=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='records')
    temp_result_file_path = models.CharField(max_length=255, null=True, blank=True)
    start_time = models.DateTimeField(null=False, blank=False)
    end_time = models.DateTimeField(null=True, blank=True)
    exec_time = models.CharField(null=True, blank=False, max_length=64)
    status = models.CharField(max_length=50, choices=[
        ('pending', '待处理'),
        ('processing', '处理中'),
        ('completed', '已完成'),
        ('failed', '失败'),
    ], default='pending')

    class Meta:
        db_table = 'records'


#
class TuningModels(models.Model):
    # 存储调优参数
    tuning_id = models.UUIDField(unique=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='tuning_model_user')
    tuning_model = models.CharField(max_length=64, null=False, blank=False, verbose_name='调优模型')
    start_time = models.DateTimeField(null=False, blank=False)
    end_time = models.DateTimeField(null=True, blank=True)

    # 用户设置的调优参数
    lr = models.FloatField(null=False, blank=False, verbose_name='学习率')
    wd = models.FloatField(null=False, blank=False, verbose_name='权重衰减')
    batch_size = models.IntegerField(null=False, blank=False, verbose_name='批量大小')
    num_epochs = models.IntegerField(null=False, blank=False, verbose_name='训练轮数')
    alpha = models.IntegerField(null=False, blank=False, verbose_name='Alpha')
    temperature = models.IntegerField(null=False, blank=False, verbose_name='Temperature ')

    # 训练结果
    accuracy = models.FloatField(null=True, blank=True, verbose_name='验证集准确率')
    loss = models.FloatField(null=True, blank=True, verbose_name='损失值')
    test_accuracy = models.FloatField(null=True, blank=True, verbose_name='测试集准确率')
    # 创建时间
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')

    class Meta:
        db_table = 'tuning_models'


class IPAddressRule(models.Model):
    IP_RULE_CHOICES = [
        ('white', 'Whitelist'),
        ('black', 'Blacklist')
    ]

    ip_address = models.CharField(max_length=100, unique=True)
    rule_type = models.CharField(max_length=20, choices=IP_RULE_CHOICES)
    description = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)  # 更新数据会自动更新时间

    def __str__(self):
        return f"{self.rule_type}: {self.ip_address}"
