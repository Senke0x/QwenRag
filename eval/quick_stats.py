#!/usr/bin/env python3
"""
快速代码统计脚本

简化版本的代码评估，用于快速获取基本统计信息
"""

import os
import subprocess
from pathlib import Path


def count_lines(file_path):
    """统计文件行数"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        total = len(lines)
        code = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
        comments = sum(1 for line in lines if line.strip().startswith('#'))
        blank = sum(1 for line in lines if not line.strip())
        
        return total, code, comments, blank
    except:
        return 0, 0, 0, 0


def scan_python_files(directory, pattern="**/*.py"):
    """扫描Python文件"""
    total_files = 0
    total_lines = 0
    total_code = 0
    total_comments = 0
    total_blank = 0
    
    for file_path in Path(directory).glob(pattern):
        if '__pycache__' in str(file_path) or '.git' in str(file_path):
            continue
            
        lines, code, comments, blank = count_lines(file_path)
        total_files += 1
        total_lines += lines
        total_code += code
        total_comments += comments
        total_blank += blank
    
    return total_files, total_lines, total_code, total_comments, total_blank


def get_test_coverage():
    """获取测试覆盖率"""
    try:
        result = subprocess.run([
            'python3', '-m', 'pytest', 'tests/', '--cov=.', '--cov-report=term-missing', '-q'
        ], capture_output=True, text=True, timeout=60)
        
        # 从输出中提取覆盖率
        for line in result.stdout.split('\n'):
            if 'TOTAL' in line and '%' in line:
                parts = line.split()
                for part in parts:
                    if '%' in part:
                        return part.replace('%', '')
        return "N/A"
    except:
        return "N/A"


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("📊 QwenRag 快速代码统计")
    print("=" * 40)
    
    # 源代码统计
    source_dirs = ['processors', 'vector_store', 'schemas', 'utils']
    source_files = 0
    source_lines = 0
    source_code = 0
    
    for dir_name in source_dirs:
        if os.path.exists(dir_name):
            files, lines, code, comments, blank = scan_python_files(dir_name)
            source_files += files
            source_lines += lines
            source_code += code
    
    # 配置文件
    config_files = ['config.py', 'main_index.py', 'main_search.py', 'demo.py']
    for config_file in config_files:
        if os.path.exists(config_file):
            lines, code, comments, blank = count_lines(config_file)
            source_files += 1
            source_lines += lines
            source_code += code
    
    # 测试代码统计
    test_files, test_lines, test_code, test_comments, test_blank = scan_python_files('tests')
    
    # 覆盖率
    print("⏳ 获取测试覆盖率...")
    coverage = get_test_coverage()
    
    # 输出结果
    print(f"\n📁 文件统计:")
    print(f"   源代码文件: {source_files}")
    print(f"   测试文件: {test_files}")
    print(f"   总文件: {source_files + test_files}")
    
    print(f"\n📝 代码行数:")
    print(f"   源代码: {source_code:,} 行")
    print(f"   测试代码: {test_code:,} 行")
    print(f"   总代码: {source_code + test_code:,} 行")
    
    print(f"\n📈 质量指标:")
    print(f"   测试覆盖率: {coverage}%")
    print(f"   测试/代码比: {test_code/source_code:.2f}" if source_code > 0 else "   测试/代码比: N/A")
    
    print(f"\n✨ 项目规模: {'小型' if source_code < 2000 else '中型' if source_code < 10000 else '大型'}")


if __name__ == "__main__":
    main()