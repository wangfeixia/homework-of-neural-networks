import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 数据处理（包括相关性分析与PCA分析）
def load_and_analyze_data(test_size=0.1, random_state=42, use_pca=False, n_components=None):
    data = pd.read_csv('BostonHousingData.csv')
   # data = pd.read_excel("BostonHousingData.xlsx")
    # 相关性分析
    plt.figure(figsize=(12, 10))
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')  # 热力图
    plt.title('Feature Correlation Matrix')
    plt.show()

    # 数据处理
    x_data = data.iloc[:, :13].values
    y_data = data.MEDV.values.reshape(-1, 1)
    feature_names = data.columns[:13]
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)

    # 3. PCA分析与可视化
    if use_pca:
        pca = PCA(n_components=n_components)
        x_pca = pca.fit_transform(x_data)

        print("\n=== 主成分特征权重 ===")
        components_df = pd.DataFrame(
            pca.components_,
            columns=feature_names,
            index=[f'PC{i + 1}' for i in range(pca.n_components_)]
        )
        print(components_df.round(3))

        print("\n=== 各主成分主要特征 ===")
        for i in range(pca.n_components_):
            print(f"\nPC{i + 1}（解释方差：{pca.explained_variance_ratio_[i]:.2%}）")
            top_features = components_df.iloc[i].abs().nlargest(3)
            print(top_features.to_string(float_format="%.3f"))

        # 解释方差比例（成分占比）
        plt.figure(figsize=(10, 6))
        plt.bar(range(pca.n_components_), pca.explained_variance_ratio_)
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        plt.title('PCA Explained Variance Ratio')
        plt.show()

        # 累计解释方差
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Cumulative Explained Variance')
        plt.grid(True)
        plt.show()

        # 主成分散点图
        if n_components >= 2:
            plt.figure(figsize=(10, 6))
            plt.scatter(x_pca[:, 0], x_pca[:, 1], c=data['MEDV'], cmap='viridis')
            plt.colorbar(label='MEDV')
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            plt.title('PCA Components Colored by MEDV')
            plt.show()

        # 使用PCA转换后的数据
        x_data = x_pca

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=test_size, random_state=random_state
    )

    # 转换为PyTorch张量
    xt_train = torch.FloatTensor(x_train)
    xt_test = torch.FloatTensor(x_test)
    yt_train = torch.FloatTensor(y_train)
    yt_test = torch.FloatTensor(y_test)

    return (xt_train, yt_train), (xt_test, yt_test), scaler, feature_names

#数据加载
(xt_train, yt_train), (xt_test, yt_test), scaler, feature_names = load_and_analyze_data(use_pca=True, n_components=13)

train_dataset = Data.TensorDataset(xt_train, yt_train)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, drop_last=True)

# 模型定义
class Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.05),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# 模型初始化打印模型
model = Model(input_dim=xt_train.shape[1])
print(model)

def train_model(model, train_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
    criterion = nn.MSELoss()  # 适用于回归问题
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.9)

    model.train()
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)

        if epoch % 15 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
    return model, loss_history

# 评估函数
def evaluate_model(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(x_test).numpy()
        y_test = y_test.numpy()

        # 计算评估指标
        mse = mean_squared_error(y_test, predictions)  #
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)  #
        r2 = r2_score(y_test, predictions)  #

        # 打印评估结果
        print("\n" + "—" * 50)
        print("Model Evaluation Metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        print("—" * 50 + "\n")

        # 绘制预测结果
        plt.figure(figsize=(12, 6))

        # 实际vs预测散点图
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, predictions, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.title('Actual vs Predicted Prices')
        plt.grid(True)

        # 残差图
        plt.subplot(1, 2, 2)
        residuals = y_test - predictions
        plt.scatter(predictions, residuals, alpha=0.6)
        plt.hlines(0, predictions.min(), predictions.max(), colors='k', linestyles='dashed')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Analysis')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

#训练模型
trained_model, loss_history = train_model(model, train_loader, epochs=450)
# 绘制损失函数
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

evaluation_results = evaluate_model(trained_model, xt_test, yt_test)

# 保存模型
torch.save({
    'model_state_dict': trained_model.state_dict(), 'scaler': scaler, 'feature_names': feature_names,
    'evaluation_metrics': evaluation_results
}, "boston_housing_model.pth")