import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

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
    accuracy = 0
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
    return correct / total


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


def apply_model_on_test_file_single(test_file_path):
    # 初始化计数器
    normal_count = 0
    abnormal_count = 0

    # 处理 outcome 列并进行计数
    def process_outcome(x):
        nonlocal normal_count, abnormal_count
        if x == 'normal':
            normal_count += 1
            return 0  # normal 对应 0
        else:
            abnormal_count += 1
            return 1  # abnormal 对应 1

    # 假设模型已经训练完毕，路径为 ./model/best_student_model.pth
    model_path = './model/best_student_model.pth'

    # 1. 加载数据集和预处理
    data_train_df = load_dataSet("main/DL/data/KDDTrain+.txt")
    # data_train_df = load_dataSet("./data/KDDTrain+.txt")
    data_test_df = load_dataSet(test_file_path)

    train_features, train_labels, test_features, test_labels = preprocess_data(data_train_df, data_test_df)

    data_test_df['outcome'] = data_test_df['outcome'].apply(process_outcome)

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
    accuracy = test_model(test_iter, student_net)
    return accuracy, normal_count, abnormal_count

# accuracy, normal_count, abnormal_count = apply_model_on_test_file_single("./data/KDDTrain+.txt")  # 数据测试