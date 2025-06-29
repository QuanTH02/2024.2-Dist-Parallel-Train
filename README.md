# Fine-tuning Language Models Project

## Mục tiêu
So sánh hiệu suất training giữa các kỹ thuật khác nhau:
- Single GPU vs Multi-GPU
- DeepSpeed ZeRO vs PyTorch FSDP  
- DeepSpeed Pipeline vs PyTorch Pipeline

## Cài đặt
```bash
pip install -r requirements.txt
```

## Cấu trúc Project
- `01_single_vs_multi_gpu.ipynb` - Single vs Multi-GPU comparison
- `02_deepspeed_vs_fsdp.ipynb` - DeepSpeed ZeRO vs PyTorch FSDP
- `03_pipeline_parallelism.ipynb` - Pipeline parallelism comparison
- `04_summary_report.ipynb` - Summary and final report
- `utils.py` - Utility functions
- `requirements.txt` - Dependencies

## Models được test
- **Small**: facebook/opt-iml-1.3b (1.3B parameters)
- **Large**: facebook/opt-6.7b (6.7B parameters)

## Kỹ thuật được so sánh
1. **Single GPU vs Multi-GPU**
   - Single GPU baseline
   - DataParallel
   - DistributedDataParallel

2. **DeepSpeed ZeRO vs PyTorch FSDP**
   - ZeRO-1, ZeRO-2, ZeRO-3
   - FSDP với/không CPU offload

3. **Pipeline Parallelism**
   - DeepSpeed Pipeline
   - PyTorch Pipeline

## Hướng dẫn sử dụng
1. Chạy notebooks theo thứ tự từ 01 đến 04
2. Đảm bảo có ít nhất 2 GPU
3. Xem kết quả trong `final_report.md`

## Output Files
- `performance_summary.csv` - Performance data
- `final_report.md` - Final report
- `*.png` - Performance charts
- `*_metrics.json` - Individual metrics

## References
- [PyTorch FSDP](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)
- [DeepSpeed Pipeline](https://www.deepspeed.ai/tutorials/pipeline/)

## License

MIT License - Feel free to use and modify for your projects.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

**Lưu ý**: Đảm bảo bạn có đủ GPU memory và computational resources để chạy các thí nghiệm này. Các large models có thể yêu cầu 40GB+ GPU memory. 