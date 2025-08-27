# 📄 保存配置到文件
import os
ap_id = [1,2]  # [1,2,3,4]   # 1代表AP1

def save_config_to_file(cfg, save_dir, dataset_fname, device, base_AP, target_AP):
    """将配置参数保存到 JSON 文件中"""
    import json
    from datetime import datetime
    
    config_dict = {
        "experiment_info": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_path": dataset_fname,
            "checkpoint_dir": save_dir,
            "device": str(device),
            "base_AP": base_AP,
            "target_AP": target_AP,
            'environment': cfg.training.args_env,
            'ap_id': cfg.training.ap_id
        },
        "training": {
            "batch_size": cfg.training.batch_size,
            "n_iters": cfg.training.n_iters,
            "display_rate": cfg.training.display_rate,
            "train_split": cfg.training.train_split,
            "val_split": cfg.training.val_split
        },
        "optimizer": {
            "lr": cfg.optimizer.lr,
            "weight_decay": cfg.optimizer.weight_decay,
            "scheduler_patience": cfg.optimizer.scheduler_patience,
            "scheduler_factor": cfg.optimizer.scheduler_factor,
            "min_lr": cfg.optimizer.min_lr
        },
        "randomseed": {
            "dataset_seed": cfg.randomseed.dataset_seed,
            "torch_seed": cfg.randomseed.torch_seed,
            "npy_seed": cfg.randomseed.npy_seed
        }
    }
    
    config_path = os.path.join(save_dir, 'training_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)
    
    print(f"Training configuration saved to: {config_path}")
    return config_path







def update_training_progress(cfg, save_dir, current_iter, train_loss, val_loss, train_snr, val_snr, current_lr):
    """更新配置文件，记录训练进度"""
    import json
    from datetime import datetime
    
    config_path = os.path.join(save_dir, 'training_config.json')
    
    # 读取现有配置
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
    else:
        config_dict = {}
    
    # 更新训练进度信息
    config_dict["training_progress"] = {
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_iteration": current_iter,
        "total_iterations": cfg.training.n_iters,
        "progress_percentage": round((current_iter / cfg.training.n_iters) * 100, 2),
        "latest_train_loss": train_loss,
        "latest_val_loss": val_loss,
        "latest_train_snr": train_snr,
        "latest_val_snr": val_snr,
        "current_learning_rate": current_lr,
        "checkpoint_saved": f"iter_{current_iter}.pt"
    }
    
    # 保存更新后的配置
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)