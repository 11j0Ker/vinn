import os

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from imblearn.over_sampling import ADASYN, SMOTE
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataSet(path):
    # 读取数据集
    df_0 = pd.read_csv(path)
    df = df_0.copy()

    # 为数据集添加列名称
    columns = (
        ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
         'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
         'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
         'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
         'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
         'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
         'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
         'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'outcome', 'level'])
    df.columns = columns
    return df


def preprocess_data(train_df, test_df, overSamplingValue):
    train_df = train_df.copy()
    test_df = test_df.copy()

    # 1. 合并训练集和测试集的标签
    combined_outcome = pd.concat([train_df['outcome'], test_df['outcome']], axis=0)

    # 2. 对合并后的标签进行编码
    le = LabelEncoder()
    le.fit(combined_outcome)  # 训练标签编码器，适应训练集和测试集中的所有标签

    # 3. 对训练集和测试集分别应用编码
    train_df['outcome'] = le.transform(train_df['outcome'])  # 对训练集标签编码
    test_df['outcome'] = le.transform(test_df['outcome'])  # 对测试集标签编码

    # 2. 对类别列进行独热编码
    train_df = pd.get_dummies(train_df, columns=['protocol_type', 'service', 'flag'], drop_first=True, dtype=int)
    test_df = pd.get_dummies(test_df, columns=['protocol_type', 'service', 'flag'], drop_first=True, dtype=int)

    # 3. 确保训练集和测试集有相同的列 (对齐列)
    train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)

    # 4. 特征和目标变量分离
    X_train_df = train_df.drop(columns=['outcome', 'level'])  # 训练集特征，去掉 'outcome' 和 'level'
    y_train = train_df['outcome']  # 训练集标签

    X_test_df = test_df.drop(columns=['outcome', 'level'])  # 测试集特征，去掉 'outcome' 和 'level'
    y_test = test_df['outcome']  # 测试集标签

    # 5. 标准化：使用训练集的均值和标准差对训练集和测试集进行标准化
    X_train = X_train_df.apply(lambda x: (x - x.mean()) / (x.std())).fillna(0)
    X_test = X_test_df.apply(lambda x: (x - x.mean()) / (x.std())).fillna(0)

    # 6. 返回标准化后的特征和标签
    return X_train, y_train, X_test, y_test


# 将数据集加载为PyTorch data iterator.
def load_array(data_arrays, batch_size, is_Train=True):
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_Train)


# 定义损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)


class InceptionModule(nn.Module):
    def __init__(self, in_channels, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
        super(InceptionModule, self).__init__()
        # 1x1 卷积分支
        self.conv1 = nn.Conv1d(in_channels, f1, kernel_size=1, padding='same')

        # 1x1 -> 3x3 卷积分支
        self.conv3_1 = nn.Conv1d(in_channels, f2_in, kernel_size=1, padding='same')
        self.conv3_2 = nn.Conv1d(f2_in, f2_out, kernel_size=3, padding='same')

        # 1x1 -> 5x5 卷积分支
        self.conv5_1 = nn.Conv1d(in_channels, f3_in, kernel_size=1, padding='same')
        self.conv5_2 = nn.Conv1d(f3_in, f3_out, kernel_size=5, padding='same')

        # MaxPool -> 1x1 卷积分支
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_channels, f4_out, kernel_size=1, padding='same')

        # 调整通道数的1x1卷积
        self.conv3_adjust = nn.Conv1d(f2_out, f4_out, kernel_size=1, padding='same')
        self.conv5_adjust = nn.Conv1d(f3_out, f4_out, kernel_size=1, padding='same')
        self.conv1_adjust = nn.Conv1d(f1, f4_out, kernel_size=1, padding='same')

    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        conv1 = F.relu(self.conv1_adjust(conv1))

        conv3 = F.relu(self.conv3_1(x))
        conv3 = F.relu(self.conv3_2(conv3))
        conv3 = F.relu(self.conv3_adjust(conv3))

        conv5 = F.relu(self.conv5_1(x))
        conv5 = F.relu(self.conv5_2(conv5))
        conv5 = F.relu(self.conv5_adjust(conv5))

        pool = self.pool(x)
        pool = F.relu(self.pool_conv(pool))

        return conv1 + conv3 + conv5 + pool


class TeacherModel(nn.Module):
    def __init__(self, num_classes):
        super(TeacherModel, self).__init__()

        # Inception module
        self.inception = InceptionModule(1, 64, 128, 128, 32, 32, 32)

        # Conv1D layer
        self.conv1d = nn.Conv1d(32, 128, kernel_size=64, padding='same')
        self.pool = nn.MaxPool1d(10)
        self.batchnorm = nn.BatchNorm1d(128)

        # GRU layer
        self.gru = nn.GRU(128, 256, batch_first=True, bidirectional=True, dropout=0.5)

        # Attention mechanism
        self.attn = nn.MultiheadAttention(embed_dim=256 * 2, num_heads=8, batch_first=True)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, x):
        # Pass through inception module
        x = x.unsqueeze(1)
        x = self.inception(x)

        # Conv1D, max pooling, and batch normalization
        x = F.relu(self.conv1d(x))
        x = self.pool(x)
        x = self.batchnorm(x)

        # GRU layer
        x, _ = self.gru(x.permute(0, 2, 1))  # GRU expects (batch, seq_len, input_size)

        # Attention mechanism
        x, _ = self.attn(x, x, x)

        # Global average pooling
        x = self.global_avg_pool(x.permute(0, 2, 1))  # Convert back to (batch, seq_len, channels)

        # Flatten and pass through dropout and fully connected layer
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return F.sigmoid(x)


class StudentModel(nn.Module):
    def __init__(self, num_classes):
        super(StudentModel, self).__init__()

        # Simplified Inception module with fewer filters
        self.inception = InceptionModule(1, 32, 64, 64, 16, 16, 16)

        # Simplified Conv1D layer
        self.conv1d = nn.Conv1d(16, 64, kernel_size=32, padding='same')
        self.pool = nn.MaxPool1d(5)
        self.batchnorm = nn.BatchNorm1d(64)

        # Simplified GRU layer
        self.gru = nn.GRU(64, 128, batch_first=True, bidirectional=True, dropout=0.3)

        # Simplified Attention mechanism
        self.attn = nn.MultiheadAttention(embed_dim=128 * 2, num_heads=4, batch_first=True)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Simplified Fully connected layers
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        # Pass through inception module
        x = self.inception(x)

        # Conv1D, max pooling, and batch normalization
        x = F.relu(self.conv1d(x))
        x = self.pool(x)
        x = self.batchnorm(x)

        # GRU layer
        x, _ = self.gru(x.permute(0, 2, 1))  # GRU expects (batch, seq_len, input_size)

        # Attention mechanism
        x, _ = self.attn(x, x, x)

        # Global average pooling
        x = self.global_avg_pool(x.permute(0, 2, 1))  # Convert back to (batch, seq_len, channels)

        # Flatten and pass through dropout and fully connected layer
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return F.sigmoid(x)


def distillation_loss(student_output, teacher_output, target, alpha, temperature):
    hard_loss = nn.CrossEntropyLoss()(student_output, target)  # 计算硬标签损失
    soft_loss = nn.KLDivLoss()(torch.log_softmax(student_output / temperature, dim=1),
                               torch.softmax(teacher_output / temperature, dim=1))  # 计算软标签损失
    return alpha * hard_loss + (1 - alpha) * soft_loss * (temperature ** 2)


def reshape_for_lstm(X):
    # 将数据从 (batch_size, input_size) 转换为 (batch_size, seq_length, input_size)
    return X.unsqueeze(1)  # 在第二维插入一个维度，使得 seq_length = 1


# 测试模型 测试模型的准确率、召回率、F1-score
def test_model(test_iter, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_iter:
            inputs = inputs.to(device)
            labels = labels.to(device).long()  # 标签转为long类型
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # 获取最大概率的类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))
    return 100 * correct / total


def train_with_distillation(student_model, teacher_model, train_iter, valid_iter, num_epochs, lr, wd, n_valid_size,
                            alpha, temperature):
    accuracy, loss = 0, 0
    trainer = torch.optim.Adam(student_model.parameters(), lr=lr, weight_decay=wd)
    numbers_batch = len(train_iter)
    student_model = student_model.to(device)  # 将学生模型移到设备上
    teacher_model = teacher_model.to(device)  # 将教师模型移到设备上
    teacher_model.eval()  # 教师模型设为评估模式

    best_accuracy = 0.0  # 记录最佳准确率
    best_epoch = 0  # 用于跟踪哪个 epoch 得到的模型最好
    for epoch in range(num_epochs):
        print(f"-------第 {epoch + 1} 轮训练开始-------")
        student_model.train()  # 学生模型设为训练模式
        total_accuracy = 0
        for i, (features, labels) in enumerate(train_iter):
            features = features.to(device)
            labels = labels.to(device).long()  # 标签转为long类型用于CrossEntropyLoss

            # 训练学生模型
            trainer.zero_grad()
            student_outputs = student_model(features)  # 学生模型预测
            with torch.no_grad():
                teacher_outputs = teacher_model(features)  # 教师模型预测

            # 计算蒸馏损失
            loss = distillation_loss(student_outputs, teacher_outputs, labels, alpha, temperature)
            loss.backward()  # 反向传播
            trainer.step()  # 更新权重

            if (i + 1) % (numbers_batch // 5) == 0 or i == numbers_batch - 1:
                print(f'epoch {epoch + 1}, iter {i + 1}: train loss {abs(loss.item()):.3f}')

        # 验证集
        total_valid_loss = 0
        valid_loss = 0
        if valid_iter is not None:
            student_model.eval()  # 学生模型设为评估模式
            with torch.no_grad():
                for X, y in valid_iter:
                    X = X.to(device)
                    targets = y.to(device).long()
                    output = student_model(X)
                    loss = loss_fn(output, targets)  # 使用普通的交叉熵损失计算验证损失
                    total_valid_loss = total_valid_loss + loss.item()
                    valid_loss = loss.item()
                    _, predicted = torch.max(output, 1)
                    accuracy = (predicted == targets).sum().item()
                    total_accuracy += accuracy
            accuracy = float(total_accuracy / n_valid_size)
            # loss = total_valid_loss
            loss = valid_loss
            print("整体验证集上的Loss: {}".format(total_valid_loss))
            print("整体验证集上的正确率: {}".format(total_accuracy / n_valid_size))

            if epoch == num_epochs - 1:
                if not os.path.exists('./model'):
                    os.mkdir('./model')
                torch.save(student_model.state_dict(), './model/best_student_model_duofenlei.pth')
    return accuracy, loss


# class SimpleLSTM(nn.Module):
#     def __init__(self, input_size, num_classes, hidden_size=64, num_layers=2):
#         super(SimpleLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)  # 输出层，修改为 num_classes
#         self.softmax = nn.Softmax(dim=1)  # Softmax 激活函数用于多分类
#
#     def forward(self, x):
#         # LSTM 期望输入的维度为 (batch_size, seq_length, input_size)
#         # 这里 seq_length 是 1，因为我们的数据集是每个样本只有一个时间步
#         x, _ = self.lstm(x)  # LSTM 层输出
#         x = x[:, -1, :]  # 取最后一个时间步的输出（LSTM 输出的是整个序列的隐藏状态）
#         x = self.fc(x)  # 全连接层
#         x = self.softmax(x)  # Softmax 激活
#         return x

# 20, 8e-5, 6e-6, 256, 0.5, 2.0
def main(num_epochs=20, lr=8e-5, wd=6e-6, batch_size=256, alpha=0.5, temperature=2.0, overSamplingValue=0):
    print(num_epochs, lr, wd, batch_size, alpha, temperature, overSamplingValue)
    # 1. 载入数据集和预处理
    data_train_df = load_dataSet("main/DL/data/KDDTrain+.txt")
    # data_train_df = load_dataSet("./data/KDDTrain+.txt")
    data_test_df = load_dataSet("main/DL/data/KDDTest+.txt")
    train_features, train_labels, test_features, test_labels = preprocess_data(data_train_df, data_test_df,
                                                                               overSamplingValue)
    # 计算一下样本标签的类别数量
    num_classes = len(test_labels.unique())
    n_train = train_features.shape[0]  # 训练集样本数
    feature_type_count = train_features.shape[1]  # 特征数量

    train_features_resampled = torch.tensor(train_features[:n_train].values, dtype=torch.float32)
    # 训练和测试标签转换为一维long类型
    train_labels_resampled = torch.tensor(train_labels.values, dtype=torch.long)  # 一维的long类型标签

    n_test = test_features.shape[0]  # 训练集样本数
    test_features = torch.tensor(test_features[:n_test].values, dtype=torch.float32)
    test_labels = torch.tensor(test_labels.values, dtype=torch.long)  # 一维的long类型标签

    # 5. 划分训练集和验证集（80% 训练集，20% 验证集）
    valid_ratio = 0.2
    n_valid = int(train_features_resampled.shape[0] * valid_ratio)

    valid_features = train_features_resampled[:n_valid]
    valid_labels = train_labels_resampled[:n_valid]
    train_features_resampled = train_features_resampled[n_valid:]
    train_labels_resampled = train_labels_resampled[n_valid:]

    # 6. 使用 DataLoader 加载训练集和验证集
    train_iter = load_array((train_features_resampled, train_labels_resampled), batch_size)
    valid_iter = load_array((valid_features, valid_labels), batch_size)
    test_iter = load_array((test_features, test_labels), batch_size, False)

    # 7. 定义和训练模型
    teacher_model = TeacherModel(num_classes=num_classes)  # 定义教师模型
    student_model = StudentModel(num_classes=num_classes)  # 定义学生模型
    # 6. 调用训练函数，训练学生模型
    accuracy, loss = train_with_distillation(student_model, teacher_model, train_iter, valid_iter, num_epochs, lr, wd,
                                             len(valid_labels),
                                             alpha, temperature)

    # 9. 评估学生模型
    test_accuracy = test_model(test_iter, student_model)
    return accuracy, loss, test_accuracy
