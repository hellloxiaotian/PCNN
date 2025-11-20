import torch
import torchvision.models as models
from thop import profile
from network.models import *

def measure_inference_time(model, input_shape, device, warmup_runs=10, test_runs=100):
    """测量模型推理时间（含预热步骤）"""
    # 准备输入数据
    input_tensor = torch.randn(*input_shape, device=device)
    
    # 初始化CUDA事件
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # GPU预热
    print(f"开始GPU预热（{warmup_runs}次）...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            model(input_tensor)
    torch.cuda.synchronize()
    print("GPU预热完成，开始正式测试...")
    
    # 正式测试
    times = []
    with torch.no_grad():
        for _ in range(test_runs):
            start_event.record()
            model(input_tensor)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            times.append(elapsed_ms)
    
    # 计算统计结果
    avg_time_ms = sum(times) / test_runs
    std_time_ms = torch.tensor(times).std().item()
    return avg_time_ms, std_time_ms


def calculate_flops_params(model, input_shape):
    """计算模型的FLOPs和参数数量"""
    # 构造输入张量（CPU上计算，不影响结果）
    input_tensor = torch.randn(*input_shape)
    
    # 统计计算量和参数
    flops, params = profile(
        model=model,
        inputs=(input_tensor,),
        verbose=False  # 如需查看每层细节可改为True
    )
    
    # 格式化输出
    def format_num(num):
        if num >= 1e9:
            return f"{num/1e9:.2f} G"
        elif num >= 1e6:
            return f"{num/1e6:.2f} M"
        elif num >= 1e3:
            return f"{num/1e3:.2f} K"
        return f"{num:.2f}"
    
    return format_num(flops), format_num(params)


if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无GPU，无法执行测试")
    
    # 模型配置
    # model = models.resnet50(pretrained=False).eval().to(device)
    # model = Model_6(num_class=7, device=device).eval()
    model = PCNN().eval().to(device)
    input_shape = (1, 3, 224, 224)  # (batch_size, channels, height, width)
    test_runs = 50  # 推理时间测试次数
    
    # 1. 计算量和参数统计
    print("开始计算模型计算量和参数...")
    flops, params = calculate_flops_params(model.cpu(), input_shape)  # 计算时临时移到CPU
    model.to(device)  # 移回GPU进行时间测试
    
    # 2. 推理时间测试
    print("\n开始测试推理时间...")
    avg_time, std_time = measure_inference_time(
        model=model,
        input_shape=input_shape,
        device=device,
        warmup_runs=10,
        test_runs=test_runs
    )
    
    # 输出所有结果
    print("\n===== 模型性能测试结果 =====")
    print(f"输入形状: {input_shape}")
    print(f"总计算量: {flops} MACs")
    print(f"总参数数量: {params} Params")
    print(f"\n推理时间统计（{test_runs}次）:")
    print(f"平均推理时间: {avg_time:.2f} ms")
    print(f"时间标准差: {std_time:.2f} ms（值越小越稳定）")
    print(f"单样本吞吐量: {1000 / avg_time:.2f} samples/sec")