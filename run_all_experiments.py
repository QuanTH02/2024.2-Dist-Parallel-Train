#!/usr/bin/env python3
"""
Script để chạy tất cả các thí nghiệm fine-tuning language models
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime

def check_environment():
    """Kiểm tra môi trường trước khi chạy"""
    print("🔍 Kiểm tra môi trường...")
    
    # Kiểm tra CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA không khả dụng!")
            return False
        
        gpu_count = torch.cuda.device_count()
        print(f"✅ CUDA khả dụng với {gpu_count} GPU(s)")
        
        if gpu_count < 2:
            print("⚠️  Cảnh báo: Cần ít nhất 2 GPU để chạy đầy đủ các thí nghiệm")
        
        return True
    except ImportError:
        print("❌ PyTorch chưa được cài đặt!")
        return False

def install_dependencies():
    """Cài đặt dependencies"""
    print("📦 Cài đặt dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies đã được cài đặt")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi cài đặt dependencies: {e}")
        return False

def run_notebook(notebook_path, timeout=1800):
    """Chạy notebook và trả về kết quả"""
    print(f"🚀 Chạy {notebook_path}...")
    
    try:
        # Chạy notebook bằng jupyter nbconvert
        result = subprocess.run([
            sys.executable, "-m", "jupyter", "nbconvert", 
            "--to", "notebook", 
            "--execute", 
            "--output", f"executed_{notebook_path}",
            notebook_path
        ], timeout=timeout, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {notebook_path} hoàn thành")
            return True
        else:
            print(f"❌ {notebook_path} thất bại: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {notebook_path} timeout sau {timeout}s")
        return False
    except Exception as e:
        print(f"❌ Lỗi chạy {notebook_path}: {e}")
        return False

def run_experiments():
    """Chạy tất cả các thí nghiệm"""
    print("🧪 Bắt đầu chạy các thí nghiệm...")
    
    notebooks = [
        "01_single_vs_multi_gpu.ipynb",
        "02_deepspeed_vs_fsdp.ipynb", 
        "03_pipeline_parallelism.ipynb",
        "04_summary_report.ipynb"
    ]
    
    results = {}
    start_time = time.time()
    
    for notebook in notebooks:
        if os.path.exists(notebook):
            print(f"\n{'='*50}")
            print(f"Chạy {notebook}")
            print(f"{'='*50}")
            
            success = run_notebook(notebook)
            results[notebook] = success
            
            if not success:
                print(f"⚠️  Bỏ qua các notebook tiếp theo do lỗi")
                break
        else:
            print(f"⚠️  Không tìm thấy {notebook}")
            results[notebook] = False
    
    total_time = time.time() - start_time
    
    return results, total_time

def generate_summary_report(results, total_time):
    """Tạo báo cáo tổng hợp"""
    print("\n📊 Tạo báo cáo tổng hợp...")
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_time_minutes": total_time / 60,
        "results": results,
        "success_count": sum(results.values()),
        "total_count": len(results)
    }
    
    # Lưu summary
    with open("experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Tạo báo cáo text
    report = f"""
# Báo cáo Chạy Thí nghiệm Fine-tuning Language Models

## Thông tin Tổng quan
- **Thời gian bắt đầu**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Tổng thời gian chạy**: {total_time/60:.1f} phút
- **Số thí nghiệm thành công**: {summary['success_count']}/{summary['total_count']}

## Kết quả Chi tiết

"""
    
    for notebook, success in results.items():
        status = "✅ Thành công" if success else "❌ Thất bại"
        report += f"- **{notebook}**: {status}\n"
    
    report += f"""
## Files được tạo
- `experiment_summary.json`: Dữ liệu tổng hợp
- `performance_summary.csv`: Dữ liệu performance
- `final_report.md`: Báo cáo chi tiết
- `*.png`: Biểu đồ và charts
- `*_metrics.json`: Metrics từng phương pháp

## Hướng dẫn tiếp theo
1. Xem `final_report.md` để có báo cáo chi tiết
2. Kiểm tra các file `*_metrics.json` để phân tích từng phương pháp
3. Xem các biểu đồ `*.png` để so sánh trực quan

---
*Báo cáo được tạo tự động vào {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open("experiment_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("✅ Báo cáo đã được tạo:")
    print("  - experiment_summary.json")
    print("  - experiment_report.md")

def main():
    """Main function"""
    print("🎯 Fine-tuning Language Models - Automated Experiments")
    print("=" * 60)
    
    # Kiểm tra môi trường
    if not check_environment():
        print("❌ Môi trường không phù hợp. Dừng chạy.")
        return
    
    # Cài đặt dependencies
    if not install_dependencies():
        print("❌ Không thể cài đặt dependencies. Dừng chạy.")
        return
    
    # Chạy thí nghiệm
    results, total_time = run_experiments()
    
    # Tạo báo cáo
    generate_summary_report(results, total_time)
    
    # In kết quả cuối
    print("\n" + "=" * 60)
    print("🎉 HOÀN THÀNH!")
    print("=" * 60)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"📈 Kết quả: {success_count}/{total_count} thí nghiệm thành công")
    print(f"⏱️  Tổng thời gian: {total_time/60:.1f} phút")
    
    if success_count == total_count:
        print("🎊 Tất cả thí nghiệm đã hoàn thành thành công!")
    else:
        print("⚠️  Một số thí nghiệm thất bại. Kiểm tra logs để debug.")
    
    print("\n📁 Files quan trọng:")
    print("  - experiment_report.md: Báo cáo tổng hợp")
    print("  - final_report.md: Báo cáo chi tiết")
    print("  - performance_summary.csv: Dữ liệu performance")

if __name__ == "__main__":
    main() 