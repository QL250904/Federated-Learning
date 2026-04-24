import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from dataset import prepare_datasets
import torch
import pickle
from pathlib import Path
import numpy as np
from visualize import visualize
from clients import generate_client_fn
import flwr as fl
from server import get_evaluate_fn, get_on_fit_config
@hydra.main(config_path='conf', config_name='base', version_base=None)
def main(cfg :DictConfig):
    print(OmegaConf.to_yaml(cfg))
    #prepare datasets
    train_loaders, val_loaders, test_loaders = prepare_datasets(cfg.num_clients, cfg.batch_size)
    print(len(train_loaders))
    print(len(train_loaders[0].dataset))

    # Data distribution of each clients
    # visualize(train_loaders,cfg.num_clients)
    # Define clients 
    client_fn = generate_client_fn(train_loaders,val_loaders, cfg.num_classes)

    strategy = fl.server.strategy.FedAvg(fraction_fit=0.00001, 
                                         min_fit_clients=cfg.num_clients_per_round_fit,
                                         fraction_evaluate=0.00001,
                                         min_evaluate_clients=cfg.num_clients_per_round_eval,
                                         min_available_clients=cfg.num_clients,
                                         on_fit_config_fn=get_on_fit_config(cfg.config_fit),
                                         evaluate_fn=get_evaluate_fn(test_loaders, cfg.num_classes))
    # Mô phỏng (Virtual Environment Simulation) trên Flower
    import time
    print("="*60)
    print(" BẮT ĐẦU CHẠY MÔ PHỎNG FLOWER (FL VIRTUAL MACHINE) ")
    print("="*60)
    
    start_time = time.time()
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={
            'num_cpus': 1.0,
            'num_gpus': 0.0,
        }
    )
    
    total_time = time.time() - start_time
    print(f"\n[Hoàn tất Mô Phỏng] Tổng thời gian chạy Flower Sim: {total_time:.2f}s")
    
    # Save result cho file cấu hình so sánh
    save_path = Path('./outputs/benchmark')
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Rút trích list Accuracy/Loss thuần túy từ obj lịch sử
    acc_list = [val[1] for val in history.metrics_distributed['accuracy']] if 'accuracy' in history.metrics_distributed else []
    loss_list = [val[1] for val in history.losses_distributed] if history.losses_distributed else []
    
    result = {
        'method': 'Flower Simulation (Centralized)',
        'num_clients': cfg.num_clients,
        'num_rounds': cfg.num_rounds,
        'total_time': total_time,
        'history': history,
        'accuracy': acc_list[-1] if acc_list else 0,
        'loss': loss_list[-1] if loss_list else 0,
        'accuracy_list': acc_list,
        'loss_list': loss_list
    }
    
    with open(str(save_path / 'flower_simulation_results.pkl'), 'wb') as h:
        pickle.dump(result, h, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"✅ Đã lưu kết quả Flower giả lập tại: {save_path / 'flower_simulation_results.pkl'}")

if __name__ ==  '__main__':
    main()

