import flwr as fl
from clients import FlowerClient
from dataset import prepare_datasets

# === Cấu hình ===
NUM_CLIENTS = 2
BATCH_SIZE = 32
NUM_CLASSES = 10
CLIENT_ID = 1  # <-- Client thứ 2

# === Load dữ liệu ===
trainloaders, valloaders, _ = prepare_datasets(NUM_CLIENTS, BATCH_SIZE)

# === Tạo Flower client ===
client = FlowerClient(
    trainloader=trainloaders[CLIENT_ID],
    valoader=valloaders[CLIENT_ID],
    num_classes=NUM_CLASSES,
).to_client()

# === Kết nối tới server ===
print(f"Client {CLIENT_ID} đang kết nối tới server 127.0.0.1:8080 ...")

fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=client,
)