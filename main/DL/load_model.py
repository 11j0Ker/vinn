import os

import pandas as pd
import torch
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 无cuda


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

    # print("过采样开始..")
    # # 7. 创建 ADASYN 对象进行过采样
    # adasyn = ADASYN(sampling_strategy='auto', random_state=42)
    # # 8. 对数据进行过采样
    # X_train_df, y_train = adasyn.fit_resample(X_train_df, y_train)
    # print("过采样结束..")

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
loss_fn = nn.BCELoss()
loss_fn = loss_fn.to(device)


class TeacherModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(TeacherModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


# 学生模型继承自教师模型，稍微简化一些
class StudentModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super(StudentModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


# 测试模型 测试模型的准确率、召回率、F1-score
def test_model(test_iter, model):
    # 加载模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_iter:
            inputs = inputs.to(device)
            inputs = reshape_for_lstm(inputs)
            labels = labels.to(device)
            outputs = model(inputs)
            predicted = (outputs >= 0.5).int()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))
    return 100 * correct / total


def reshape_for_lstm(X):
    return X.unsqueeze(1)  # 在第二维插入一个维度，使得 seq_length = 1


def train_model_with_distillation(student_net, teacher_net, train_iter, valid_iter, num_epochs, lr, wd, n_valid_size,
                                  alpha, temperature):
    accuracy, loss = 0, 0
    trainer = torch.optim.Adam(student_net.parameters(), lr=lr, weight_decay=wd)
    numbers_batch = len(train_iter)
    student_net = student_net.to(device)
    teacher_net = teacher_net.to(device)
    teacher_net.eval()  # 固定教师模型为评估模式
    for epoch in range(num_epochs):
        print("-------第 {} 轮训练开始-------".format(epoch + 1))
        student_net.train()
        total_accuracy = 0
        for i, (features, labels) in enumerate(train_iter):
            features = features.to(device)
            labels = labels.to(device)
            features = reshape_for_lstm(features)
            trainer.zero_grad()

            # 获取教师模型的预测
            with torch.no_grad():
                teacher_output = teacher_net(features)

            # 学生模型的预测
            student_output = student_net(features)

            # 计算蒸馏损失
            loss = distillation_loss(student_output, teacher_output, labels, alpha, temperature)
            loss.backward()
            trainer.step()

            if (i + 1) % (numbers_batch // 5) == 0 or i == numbers_batch - 1:
                print(f'epoch {epoch + 1}, iter {i + 1}: train loss {loss.item():.3f}')

        # 验证集
        total_valid_loss = 0
        if valid_iter is not None:
            student_net.eval()
            with torch.no_grad():
                for X, y in valid_iter:
                    X = X.to(device)
                    targets = y.to(device)
                    X = reshape_for_lstm(X)
                    output = student_net(X)
                    loss = nn.BCELoss()(output, targets)
                    total_valid_loss += loss.item()
                    accuracy = ((output > 0.5).float() == targets).sum()
                    total_accuracy += accuracy
            accuracy = float(total_accuracy / n_valid_size)
            loss = total_valid_loss
            print("整体验证集上的Loss: {}".format(total_valid_loss))
            print("整体验证集上的正确率: {}".format(total_accuracy / n_valid_size))
            if epoch == num_epochs - 1:
                if not os.path.exists('./model'):
                    os.mkdir('./model')
                torch.save(student_net.state_dict(), './model/best_student_model.pth')
    return accuracy, loss


def distillation_loss(student_output, teacher_output, target, alpha=0.5, temperature=2.0):
    hard_loss = nn.BCELoss()(student_output, target)
    soft_loss = nn.KLDivLoss()(torch.log_softmax(student_output / temperature, dim=1),
                               torch.softmax(teacher_output / temperature, dim=1))
    return alpha * hard_loss + (1 - alpha) * soft_loss * (temperature ** 2)


def main(num_epochs=20, lr=5e-6, wd=6e-6, batch_size=256, alpha=0.5, temperature=2.0, overSamplingValue=0):
    print(num_epochs, lr, wd, batch_size, alpha, temperature, overSamplingValue)
    # 1. 载入数据集和预处理
    data_train_df = load_dataSet("main/DL/data/KDDTrain+.txt")
    # data_train_df = load_dataSet("./data/KDDTrain+.txt")
    data_test_df = load_dataSet("main/DL/data/KDDTest+.txt")
    # data_test_df = load_dataSet("./data/KDDTest+.txt")
    train_features, train_labels, test_features, test_labels = preprocess_data(data_train_df, data_test_df)

    n_train = train_features.shape[0]  # 训练集样本数
    feature_type_count = train_features.shape[1]  # 特征数量

    train_features_resampled = torch.tensor(train_features[:n_train].values, dtype=torch.float32)
    train_labels_resampled = torch.tensor(train_labels.values, dtype=torch.float32).view(-1, 1)

    n_test = test_features.shape[0]  # 训练集样本数
    test_features = torch.tensor(test_features[:n_test].values, dtype=torch.float32)
    test_labels = torch.tensor(test_labels.values, dtype=torch.float32).view(-1, 1)

    # # 4. 随机打乱数据集划分训练集和验证集
    # train_features_resampled, train_labels_resampled = shuffle(train_features, train_labels,
    #                                                            random_state=42)

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
    teacher_net = TeacherModel(feature_type_count)
    student_net = StudentModel(feature_type_count)

    # 8. 调用训练函数
    accuracy, loss = train_model_with_distillation(student_net, teacher_net, train_iter, valid_iter, num_epochs, lr, wd,
                                                   len(valid_features), alpha, temperature)
    test_accuracy = test_model(test_iter, student_net)

    return accuracy, loss, test_accuracy
