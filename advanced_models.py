import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP_Advanced(nn.Module):
    """
    带有高级权重初始化选项的MLP模型
    """
    def __init__(self, input_dim=3 + 2, hidden_dim=128, output_dim=2, 
                 init_method='he', dropout_rate=0.0, use_batch_norm=False):
        super(MLP_Advanced, self).__init__()
        
        self.init_method = init_method
        self.use_batch_norm = use_batch_norm
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)
        self.layer5 = nn.Linear(hidden_dim, hidden_dim)
        self.layer6 = nn.Linear(hidden_dim, output_dim)
        
        # 可选的批量归一化层
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.bn3 = nn.BatchNorm1d(hidden_dim)
            self.bn4 = nn.BatchNorm1d(hidden_dim)
            self.bn5 = nn.BatchNorm1d(hidden_dim)
        
        # 可选的Dropout层
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        高级权重初始化方法
        支持多种初始化策略
        """
        print(f"🔧 使用 '{self.init_method}' 方法初始化权重...")
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if self.init_method == 'he' or self.init_method == 'kaiming':
                    # He/Kaiming初始化 - 适合ReLU激活函数
                    if module == self.layer6:  # 输出层
                        nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='linear')
                        gain = 0.5  # 对输出层使用较小的gain
                        with torch.no_grad():
                            module.weight.data *= gain
                    else:
                        nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                
                elif self.init_method == 'xavier' or self.init_method == 'glorot':
                    # Xavier/Glorot初始化 - 适合tanh/sigmoid激活函数
                    if module == self.layer6:
                        nn.init.xavier_uniform_(module.weight, gain=0.5)
                    else:
                        nn.init.xavier_uniform_(module.weight, gain=1.0)
                
                elif self.init_method == 'normal':
                    # 正态分布初始化
                    std = 0.02 if module == self.layer6 else 0.1
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                
                elif self.init_method == 'orthogonal':
                    # 正交初始化 - 有助于梯度流动
                    gain = 0.5 if module == self.layer6 else 1.0
                    nn.init.orthogonal_(module.weight, gain=gain)
                
                elif self.init_method == 'nerf_default':
                    # NeRF论文中常用的初始化方法
                    if module == self.layer6:
                        # 输出层使用零初始化
                        nn.init.constant_(module.weight, 0.0)
                    else:
                        # 隐藏层使用正态分布
                        nn.init.normal_(module.weight, mean=0.0, std=0.1)
                
                # 偏置初始化
                if module.bias is not None:
                    if self.init_method == 'nerf_default' and module == self.layer6:
                        # NeRF输出层偏置初始化为小正值
                        nn.init.constant_(module.bias, 0.1)
                    else:
                        nn.init.constant_(module.bias, 0.0)
        
        # 打印初始化信息
        self._print_weight_stats()
    
    def _print_weight_stats(self):
        """打印权重统计信息"""
        print("📊 权重初始化统计:")
        for name, param in self.named_parameters():
            if 'weight' in name:
                mean = param.data.mean().item()
                std = param.data.std().item()
                print(f"  {name}: mean={mean:.6f}, std={std:.6f}, shape={list(param.shape)}")
    
    def forward(self, STA_loc, input_cfr):
        # 连接输入
        input_x = torch.cat([STA_loc, input_cfr], dim=-1)
        x = input_x
        
        # 前向传播
        x = F.relu(self.layer1(x))
        if self.use_batch_norm:
            x = self.bn1(x)
        if self.dropout:
            x = self.dropout(x)
        
        x = F.relu(self.layer2(x))
        if self.use_batch_norm:
            x = self.bn2(x)
        if self.dropout:
            x = self.dropout(x)
        
        x = F.relu(self.layer3(x))
        if self.use_batch_norm:
            x = self.bn3(x)
        if self.dropout:
            x = self.dropout(x)
        
        x = F.relu(self.layer4(x))
        if self.use_batch_norm:
            x = self.bn4(x)
        if self.dropout:
            x = self.dropout(x)
        
        x = F.relu(self.layer5(x))
        if self.use_batch_norm:
            x = self.bn5(x)
        if self.dropout:
            x = self.dropout(x)
        
        # 输出层
        x = self.layer6(x)
        re_ch = torch.tanh(x[..., 0])
        im_ch = torch.tanh(x[..., 1])
        output = re_ch + 1j * im_ch
        return output

# 权重初始化工具函数
def initialize_model_weights(model, method='he'):
    """
    为现有模型初始化权重的工具函数
    
    Args:
        model: PyTorch模型
        method: 初始化方法 ('he', 'xavier', 'normal', 'orthogonal', 'nerf_default')
    """
    print(f"🔧 使用 '{method}' 方法初始化模型权重...")
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if method == 'he':
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            elif method == 'xavier':
                nn.init.xavier_uniform_(module.weight)
            elif method == 'normal':
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
            elif method == 'orthogonal':
                nn.init.orthogonal_(module.weight)
            elif method == 'nerf_default':
                if 'layer6' in name:  # 输出层
                    nn.init.constant_(module.weight, 0.0)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=0.1)
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    print("✅ 权重初始化完成")

if __name__ == "__main__":
    # 测试不同的初始化方法
    print("="*60)
    print("🧪 测试不同的权重初始化方法")
    print("="*60)
    
    methods = ['he', 'xavier', 'normal', 'orthogonal', 'nerf_default']
    
    for method in methods:
        print(f"\n🔬 测试 {method} 初始化:")
        model = MLP_Advanced(init_method=method)
        
        # 测试前向传播
        batch_size = 4
        sta_loc = torch.randn(batch_size, 3)
        input_cfr = torch.randn(batch_size, 2)
        
        with torch.no_grad():
            output = model(sta_loc, input_cfr)
            print(f"  输出形状: {output.shape}")
            print(f"  输出均值: {output.abs().mean().item():.6f}")
            print(f"  输出标准差: {output.abs().std().item():.6f}")
