import torch
import torch.nn as nn
from hylomodeling_t5_our import VisualEmbedding  # 导入您的视觉嵌入模块

def test_visual_embedding():
    class Config:
        feat_dim = 2048
        pos_dim = 4
        d_model = 768
        use_vis_layer_norm = True
        individual_vis_layer_norm = False
        use_vis_order_embedding = True
        layer_norm_epsilon = 1e-6
        n_images = 10

    # 初始化配置
    config = Config()

    # 初始化对象顺序嵌入
    obj_order_embedding = nn.Embedding(32200, config.d_model)

    # 初始化视觉嵌入模块
    visual_embedding = VisualEmbedding(config, obj_order_embedding)

    # 模拟输入特征
    feats = torch.randn(2, 36, config.feat_dim)  # [B, N, feat_dim]
    pos = torch.randn(2, 36, config.pos_dim)  # [B, N, pos_dim]

    # 调用视觉嵌入模块
    vis_embedding = visual_embedding(feats, pos)

    # 打印输出形状
    print(f"Visual embedding shape: {vis_embedding.shape}")  # 应为 [2, 36, 768]

# 运行测试
if __name__ == "__main__":
    test_visual_embedding()
