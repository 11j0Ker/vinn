import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import os

import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from imblearn.over_sampling import ADASYN, SMOTE
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reshape_for_lstm(X):
    # 将数据从 (batch_size, input_size) 转换为 (batch_size, seq_length, input_size)
    return X.unsqueeze(1)  # 在第二维插入一个维度，使得 seq_length = 1

def load_trained_model(model, model_path):
    # 加载保存的模型权重
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()  # 设置模型为评估模式
    return model


def test_model(test_iter, model):
    # 加载模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_iter:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            predicted = (outputs >= 0.5).int()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))

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

    # 将非正常行为标记为攻击
    df.loc[df['outcome'] == "normal", "outcome"] = 'normal'
    df.loc[df['outcome'] != 'normal', "outcome"] = 'attack'
    return df


def preprocess_data(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    # 1. 处理 'outcome' 列
    train_df['outcome'] = train_df['outcome'].apply(lambda x: 0 if x == 'normal' else 1)
    test_df['outcome'] = test_df['outcome'].apply(lambda x: 0 if x == 'normal' else 1)

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
    def __init__(self, num_classes=1):
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
    def __init__(self, input_length, num_classes=1):
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

if __name__ == '__main__':
    # 假设模型已经训练完毕，路径为 ./model/best_student_model.pth
    model_path = './model/best_student_model.pth'

    # 1. 加载数据集和预处理
    data_train_df = load_dataSet("./nsl-kdd/KDDTrain+.txt")
    data_test_df = load_dataSet("./nsl-kdd/KDDTest+.txt")
    train_features, train_labels, test_features, test_labels = preprocess_data(data_train_df, data_test_df)
    feature_type_count = train_features.shape[1]  # 特征数量
    # 2. 转换数据为 Tensor
    test_features = torch.tensor(test_features.values, dtype=torch.float32)
    test_labels = torch.tensor(test_labels.values, dtype=torch.float32).view(-1, 1)

    # 3. 创建测试集 DataLoader
    test_iter = load_array((test_features, test_labels), 256, is_Train=False)

    # 4. 初始化学生模型（StudentModel）并加载预训练权重
    student_net = StudentModel(feature_type_count)
    student_net = load_trained_model(student_net, model_path)

    # 5. 测试模型
    test_model(test_iter, student_net)
