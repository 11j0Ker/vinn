import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reshape_for_lstm(X):
    # 将数据从 (batch_size, input_size) 转换为 (batch_size, seq_length, input_size)
    return X.unsqueeze(1)  # 在第二维插入一个维度，使得 seq_length = 1


def load_label_mapping(file_path):
    label_mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            label, encoded_value = line.strip().split(': ')
            label_mapping[int(encoded_value)] = label  # 数字为键，标签为值
    return label_mapping


def load_trained_model(model, model_path):
    # 加载保存的模型权重
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()  # 设置模型为评估模式
    return model


# 测试模型的准确率、召回率、F1-score并统计类别数量
def test_model(test_iter, model):
    model.eval()  # 切换到评估模式
    correct = 0
    total = 0
    label_path = "main/DL/features/label_mapping.txt"
    label_counts = {label: 0 for label in load_label_mapping(label_path).values()}  # 统计每个类别的数量

    # 获取类别映射
    label_mapping = load_label_mapping(label_path)

    with torch.no_grad():
        for inputs, labels in test_iter:
            inputs = inputs.to(device)
            inputs = reshape_for_lstm(inputs)
            labels = labels.to(device).long()  # 标签转为long类型
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)  # 获取最大概率的类别

            for label in predicted.cpu().tolist():  # 使用 tolist() 将 tensor 转换为列表
                predicted_label = label_mapping[label]  # 获取对应的标签名
                label_counts[predicted_label] += 1

            predicted = predicted.cpu().view(-1, 1)  # 或者使用 predicted.unsqueeze(1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算准确率
    accuracy = correct / total
    print(f'Accuracy of the network on the test set: {accuracy} %')

    # 输出每个类别的预测数量
    print("Predicted counts for each class:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")
    return accuracy, label_counts


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


def preprocess_data(train_df, test_df,data_test_df_real):
    train_df = train_df.copy()
    test_df = test_df.copy()

    # 1. 合并训练集和测试集的标签
    combined_outcome = pd.concat([train_df['outcome'], test_df['outcome']], axis=0)

    # 2. 对合并后的标签进行编码
    le = LabelEncoder()
    le.fit(combined_outcome)  # 训练标签编码器，适应训练集和测试集中的所有标签

    # 3. 对训练集和测试集分别应用编码
    train_df['outcome'] = le.transform(train_df['outcome'])  # 对训练集标签编码
    # test_df['outcome'] = le.transform(test_df['outcome'])  # 对测试集标签编码
    data_test_df_real['outcome'] = le.transform(data_test_df_real['outcome'])  # 对测试集标签编码

    # 2. 对类别列进行独热编码
    train_df = pd.get_dummies(train_df, columns=['protocol_type', 'service', 'flag'], drop_first=True, dtype=int)
    # test_df = pd.get_dummies(test_df, columns=['protocol_type', 'service', 'flag'], drop_first=True, dtype=int)
    data_test_df_real = pd.get_dummies(data_test_df_real, columns=['protocol_type', 'service', 'flag'], drop_first=True, dtype=int)

    # 3. 确保训练集和测试集有相同的列 (对齐列)
    train_df, data_test_df_real = train_df.align(data_test_df_real, join='left', axis=1, fill_value=0)

    # 4. 特征和目标变量分离
    X_train_df = train_df.drop(columns=['outcome', 'level'])  # 训练集特征，去掉 'outcome' 和 'level'
    y_train = train_df['outcome']  # 训练集标签

    X_test_df = data_test_df_real.drop(columns=['outcome', 'level'])  # 测试集特征，去掉 'outcome' 和 'level'
    y_test = data_test_df_real['outcome']  # 测试集标签

    # 5. 标准化：使用训练集的均值和标准差对训练集和测试集进行标准化
    X_train = X_train_df.apply(lambda x: (x - x.mean()) / (x.std())).fillna(0)
    X_test = X_test_df.apply(lambda x: (x - x.mean()) / (x.std())).fillna(0)

    # 6. 返回标准化后的特征和标签
    return X_test, y_test


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
    def __init__(self, input_size, num_classes, hidden_size=32, num_layers=1):
        super(StudentModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # 输出改为 num_classes
        self.softmax = nn.Softmax(dim=1)  # Softmax 层用于多分类问题

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        x = self.softmax(x)
        return x


def apply_model_on_test_file_muti(test_file_path):
    # 假设模型已经训练完毕，路径为 ./model/best_student_model.pth
    model_path = './model/best_student_model_duofenlei.pth'

    # 1. 加载数据集和预处理
    data_train_df = load_dataSet("main/DL/data/KDDTrain+.txt")
    # data_train_df = load_dataSet("./data/KDDTrain+.txt")
    # data_test_df = load_dataSet("./data/KDDTest+.txt")
    data_test_df = load_dataSet("main/DL/data/KDDTest+.txt")
    data_test_df_real = load_dataSet(test_file_path)
    test_features, test_labels = preprocess_data(data_train_df, data_test_df,data_test_df_real)
    num_classes = len(test_labels.unique())
    feature_type_count = test_features.shape[1]  # 特征数量
    # 2. 转换数据为 Tensor
    test_features = torch.tensor(test_features.values, dtype=torch.float32)
    test_labels = torch.tensor(test_labels.values, dtype=torch.float32).view(-1, 1)

    # 3. 创建测试集 DataLoader
    test_iter = load_array((test_features, test_labels), 256, is_Train=False)

    # 4. 初始化学生模型（StudentModel）并加载预训练权重
    student_net = StudentModel(feature_type_count, 38)
    student_net = load_trained_model(student_net, model_path)

    # 5. 测试模型
    accuracy, label_counts = test_model(test_iter, student_net)
    return accuracy, label_counts
# apply_model_on_test_file_muti("./data/KDDTrain+.txt")