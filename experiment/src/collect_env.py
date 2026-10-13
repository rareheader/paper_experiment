"""采集本机环境配置,写入 environment_report.txt。

只依赖标准库和已安装的 torch/numpy/matplotlib/torchvision,可重复执行。
"""
import ctypes
import datetime
import os
import platform
import sys
from pathlib import Path


def total_ram_gb():
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.ullTotalPhys / 1024**3, st.ullAvailPhys / 1024**3


def cpu_name():
    """Windows 注册表里的 CPU 型号名,读不到再退回 platform.processor()。"""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except OSError:
        return platform.processor()


def main():
    lines = []
    add = lines.append
    add(f"SFLBR 复现环境报告(生成时间:{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    add("=" * 64)
    add("")
    add("[操作系统]")
    add(f"  系统: {platform.system()} {platform.release()} (build {platform.version()})")
    add(f"  Python: {sys.version.split()[0]} ({platform.architecture()[0]})")
    add("")
    add("[硬件]")
    add(f"  CPU: {cpu_name()}")
    add(f"  逻辑核: {os.cpu_count()}")
    total, avail = total_ram_gb()
    add(f"  内存: {total:.1f} GB(当前可用 {avail:.1f} GB)")
    add("")
    add("[GPU]")
    try:
        import torch
        add(f"  torch.cuda.is_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            add(f"  GPU: {torch.cuda.get_device_name(0)}")
            add(f"  CUDA: {torch.version.cuda}")
        else:
            add("  无 CUDA 设备 —— 本机所有实验在 CPU 上跑;论文全规模实验需要 NVIDIA GPU")
    except ImportError:
        add("  torch 未安装")
    add("")
    add("[关键依赖版本]")
    for pkg in ["torch", "torchvision", "numpy", "matplotlib"]:
        try:
            m = __import__(pkg)
            add(f"  {pkg}: {m.__version__}")
        except ImportError:
            add(f"  {pkg}: 未安装")
    try:
        import torch
        add(f"  torch 默认线程数: {torch.get_num_threads()}")
    except ImportError:
        pass
    add("")
    add("[数据集缓存]")
    data_root = Path(__file__).resolve().parent.parent / "data"
    if data_root.exists():
        for p in sorted(data_root.iterdir()):
            if p.is_dir():
                add(f"  {p.name}/")
            else:
                add(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
    else:
        add("  data/ 目录不存在,首次运行会自动下载")

    report = "\n".join(lines) + "\n"
    out = Path(__file__).resolve().parent.parent / "environment_report.txt"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
