import pytest
import numpy as np
import torch
import os
from rlearn_dev.application.imatation.basic.behavior_cloning import BehaviorCloning

@pytest.fixture
def bc_model():
    """创建行为克隆模型实例"""
    observation_space = {
        'state': {'shape': (10,)},
        'market': {'shape': (5,)}
    }
    action_space = {
        'action': {'shape': (3,)}
    }
    
    return BehaviorCloning(
        observation_space=observation_space,
        action_space=action_space,
        learning_rate=1e-3,
        hidden_dims=[64, 32],
        device="cpu"
    )

@pytest.fixture
def expert_data():
    """创建模拟的专家数据"""
    batch_size = 16
    return {
        'observations': {
            'state': np.random.randn(batch_size, 10),
            'market': np.random.randn(batch_size, 5)
        },
        'actions': np.random.randn(batch_size, 3)
    }

def test_initialization(bc_model):
    """测试模型初始化"""
    assert bc_model.input_dim == 15  # 10 + 5
    assert bc_model.output_dim == 3
    assert isinstance(bc_model.policy, torch.nn.Module)

def test_preprocess_observation(bc_model):
    """测试观察值预处理"""
    obs = {
        'state': np.random.randn(10),
        'market': np.random.randn(5)
    }
    processed_obs = bc_model._preprocess_observation(obs)
    assert isinstance(processed_obs, torch.Tensor)
    assert processed_obs.shape == (15,)

def test_forward(bc_model):
    """测试前向推理"""
    obs = {
        'state': np.random.randn(10),
        'market': np.random.randn(5)
    }
    action, state = bc_model.forward(obs)
    assert isinstance(action, np.ndarray)
    assert action.shape == (3,)
    assert state is None

def test_train_step(bc_model, expert_data):
    """测试训练步骤"""
    # 准备训练数据
    batch = {
        'observations': np.concatenate([
            expert_data['observations']['state'],
            expert_data['observations']['market']
        ], axis=1),
        'actions': expert_data['actions']
    }
    
    # 执行训练
    info = bc_model.train_step(batch)
    
    # 验证训练信息
    assert 'loss' in info
    assert 'step' in info
    assert info['step'] == 0

def test_save_load(bc_model, tmp_path):
    """测试模型保存和加载"""
    # 使用pytest的tmp_path进行临时文件管理
    model_path = os.path.join(tmp_path, "test_model.pth")
    
    # 保存模型
    bc_model.save(model_path)
    
    # 创建新模型实例
    new_bc = BehaviorCloning(
        observation_space={'state': {'shape': (10,)}, 'market': {'shape': (5,)}},
        action_space={'action': {'shape': (3,)}},
        device="cpu"
    )
    
    # 加载模型
    new_bc.load(model_path)
    
    # 验证参数是否相同
    for p1, p2 in zip(bc_model.policy.parameters(), new_bc.policy.parameters()):
        assert torch.all(torch.eq(p1, p2))

def test_reset(bc_model, expert_data):
    """测试重置功能"""
    # 执行一些训练步骤
    batch = {
        'observations': np.concatenate([
            expert_data['observations']['state'],
            expert_data['observations']['market']
        ], axis=1),
        'actions': expert_data['actions']
    }
    bc_model.train_step(batch)
    
    # 重置
    bc_model.reset()
    
    # 验证状态
    assert bc_model._current_step == 0
    assert len(bc_model._training_info) == 0

def test_training_info_property(bc_model, expert_data):
    """测试训练信息属性"""
    # 执行训练步骤
    batch = {
        'observations': np.concatenate([
            expert_data['observations']['state'],
            expert_data['observations']['market']
        ], axis=1),
        'actions': expert_data['actions']
    }
    
    info = bc_model.train_step(batch)
    training_info = bc_model.training_info
    
    assert training_info == info
    assert 'loss' in training_info
    assert 'step' in training_info 