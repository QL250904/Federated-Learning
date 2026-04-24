import subprocess
import time
import os
import sys

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

os.makedirs('outputs/benchmark', exist_ok=True)
log_file = open('outputs/benchmark/physical_logs.txt', 'w', encoding='utf-8')

def run_rf():
    log_file.write("="*60 + "\n")
    log_file.write(" BẮT ĐẦU CHẠY RANDOM FOREST TRONG MÔI TRƯỜNG MẠNG THỰC \n")
    log_file.write("="*60 + "\n")
    
    server_proc = subprocess.Popen([sys.executable, 'rf_server.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='ignore')
    time.sleep(2) 
    c1_proc = subprocess.Popen([sys.executable, 'rf_client1.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='ignore')
    c2_proc = subprocess.Popen([sys.executable, 'rf_client2.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='ignore')
    
    for line in server_proc.stdout:
        log_file.write(line)
        log_file.flush()
    
    server_proc.wait()
    c1_proc.wait()
    c2_proc.wait()

def run_decentralized():
    log_file.write("\n" + "="*60 + "\n")
    log_file.write(" BẮT ĐẦU CHẠY FL DECENTRALIZED TRONG MÔI TRƯỜNG MẠNG THỰC \n")
    log_file.write("="*60 + "\n")
    
    n1_proc = subprocess.Popen([sys.executable, 'decentralized_node1.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='ignore')
    time.sleep(2) 
    n2_proc = subprocess.Popen([sys.executable, 'decentralized_node2.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='ignore')
    
    for line in n1_proc.stdout:
        log_file.write(line)
        log_file.flush()
        
    n1_proc.wait()
    n2_proc.wait()

def run_fl_centralized():
    log_file.write("\n" + "="*60 + "\n")
    log_file.write(" BẮT ĐẦU CHẠY FL CENTRALIZED (MẠNG HOA) TRONG MÔI TRƯỜNG THỰC \n")
    log_file.write("="*60 + "\n")
    
    s_proc = subprocess.Popen([sys.executable, 'server.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='ignore')
    time.sleep(3) 
    c1_proc = subprocess.Popen([sys.executable, 'run_client.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='ignore')
    c2_proc = subprocess.Popen([sys.executable, 'run_client2.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='ignore')
    
    for line in s_proc.stdout:
        log_file.write(line)
        log_file.flush()
        
    s_proc.wait()
    c1_proc.wait()
    c2_proc.wait()

if __name__ == '__main__':
    run_rf()
    run_decentralized()
    run_fl_centralized()
    log_file.close()
