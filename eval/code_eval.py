#!/usr/bin/env python3
"""
QwenRag 代码评估脚本

功能:
- 统计源代码行数
- 统计测试代码行数  
- 生成测试覆盖率报告
- 输出项目代码质量指标
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import re

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class CodeStats:
    """代码统计信息"""
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    files_count: int = 0


@dataclass
class ProjectStats:
    """项目统计信息"""
    source_code: CodeStats
    test_code: CodeStats
    config_files: CodeStats
    documentation: CodeStats
    total_files: int = 0


class CodeEvaluator:
    """代码评估器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.source_dirs = ['processors', 'vector_store', 'schemas', 'utils']
        self.test_dirs = ['tests']
        self.config_files = ['config.py', 'main_index.py', 'main_search.py', 'demo.py']
        self.doc_files = ['*.md', '*.rst', '*.txt']
        self.exclude_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv', 'venv'}
        
    def count_lines_in_file(self, file_path: Path) -> CodeStats:
        """统计单个文件的行数"""
        stats = CodeStats(files_count=1)
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            stats.total_lines = len(lines)
            
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    stats.blank_lines += 1
                elif stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                    stats.comment_lines += 1
                else:
                    # 检查行内注释
                    if '#' in stripped and not stripped.startswith('"') and not stripped.startswith("'"):
                        stats.code_lines += 1  # 有代码的行
                    else:
                        stats.code_lines += 1
                        
        except Exception as e:
            print(f"警告: 读取文件失败 {file_path}: {e}")
            
        return stats
    
    def scan_directory(self, directory: Path, file_extensions: List[str] = None) -> CodeStats:
        """扫描目录并统计代码行数"""
        if file_extensions is None:
            file_extensions = ['.py']
            
        total_stats = CodeStats()
        
        if not directory.exists():
            return total_stats
            
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix in file_extensions:
                # 排除特定目录
                if any(exclude_dir in file_path.parts for exclude_dir in self.exclude_dirs):
                    continue
                    
                file_stats = self.count_lines_in_file(file_path)
                total_stats.total_lines += file_stats.total_lines
                total_stats.code_lines += file_stats.code_lines
                total_stats.comment_lines += file_stats.comment_lines
                total_stats.blank_lines += file_stats.blank_lines
                total_stats.files_count += file_stats.files_count
                
        return total_stats
    
    def get_source_code_stats(self) -> CodeStats:
        """获取源代码统计"""
        total_stats = CodeStats()
        
        for dir_name in self.source_dirs:
            dir_path = self.project_root / dir_name
            dir_stats = self.scan_directory(dir_path)
            total_stats.total_lines += dir_stats.total_lines
            total_stats.code_lines += dir_stats.code_lines
            total_stats.comment_lines += dir_stats.comment_lines
            total_stats.blank_lines += dir_stats.blank_lines
            total_stats.files_count += dir_stats.files_count
            
        # 添加配置文件
        for config_file in self.config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                file_stats = self.count_lines_in_file(config_path)
                total_stats.total_lines += file_stats.total_lines
                total_stats.code_lines += file_stats.code_lines
                total_stats.comment_lines += file_stats.comment_lines
                total_stats.blank_lines += file_stats.blank_lines
                total_stats.files_count += file_stats.files_count
                
        return total_stats
    
    def get_test_code_stats(self) -> CodeStats:
        """获取测试代码统计"""
        total_stats = CodeStats()
        
        for dir_name in self.test_dirs:
            dir_path = self.project_root / dir_name
            dir_stats = self.scan_directory(dir_path)
            total_stats.total_lines += dir_stats.total_lines
            total_stats.code_lines += dir_stats.code_lines
            total_stats.comment_lines += dir_stats.comment_lines
            total_stats.blank_lines += dir_stats.blank_lines
            total_stats.files_count += dir_stats.files_count
            
        return total_stats
    
    def get_documentation_stats(self) -> CodeStats:
        """获取文档统计"""
        total_stats = CodeStats()
        doc_extensions = ['.md', '.rst', '.txt']
        
        for file_path in self.project_root.rglob('*'):
            if file_path.is_file() and file_path.suffix in doc_extensions:
                # 排除特定目录
                if any(exclude_dir in file_path.parts for exclude_dir in self.exclude_dirs):
                    continue
                    
                file_stats = self.count_lines_in_file(file_path)
                total_stats.total_lines += file_stats.total_lines
                total_stats.code_lines += file_stats.code_lines
                total_stats.comment_lines += file_stats.comment_lines
                total_stats.blank_lines += file_stats.blank_lines
                total_stats.files_count += file_stats.files_count
                
        return total_stats
    
    def run_coverage_analysis(self) -> Dict:
        """运行测试覆盖率分析"""
        coverage_data = {
            'coverage_percentage': 0,
            'total_statements': 0,
            'covered_statements': 0,
            'missing_lines': 0,
            'files_coverage': {},
            'error': None
        }
        
        try:
            # 检查是否安装了coverage
            subprocess.run(['python3', '-c', 'import coverage'], 
                         check=True, capture_output=True, cwd=self.project_root)
            
            # 运行带覆盖率的测试
            print("正在运行测试覆盖率分析...")
            result = subprocess.run([
                'python3', '-m', 'pytest', 'tests/', 
                '--cov=.', '--cov-report=json', '--cov-report=term-missing', '-q'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                # 读取coverage.json文件
                coverage_file = self.project_root / 'coverage.json'
                if coverage_file.exists():
                    with open(coverage_file, 'r') as f:
                        cov_data = json.load(f)
                    
                    # 解析覆盖率数据
                    totals = cov_data.get('totals', {})
                    coverage_data['coverage_percentage'] = totals.get('percent_covered', 0)
                    coverage_data['total_statements'] = totals.get('num_statements', 0)
                    coverage_data['covered_statements'] = totals.get('covered_lines', 0)
                    coverage_data['missing_lines'] = totals.get('missing_lines', 0)
                    
                    # 文件级别的覆盖率
                    files = cov_data.get('files', {})
                    for file_path, file_data in files.items():
                        if not any(exclude in file_path for exclude in ['test_', '__pycache__', '.git']):
                            coverage_data['files_coverage'][file_path] = {
                                'coverage': file_data.get('summary', {}).get('percent_covered', 0),
                                'statements': file_data.get('summary', {}).get('num_statements', 0),
                                'missing': len(file_data.get('missing_lines', []))
                            }
                    
                    print(f"测试覆盖率分析完成: {coverage_data['coverage_percentage']:.1f}%")
                else:
                    coverage_data['error'] = "未找到coverage.json文件"
            else:
                coverage_data['error'] = f"测试执行失败: {result.stderr}"
                
        except subprocess.CalledProcessError:
            coverage_data['error'] = "coverage模块未安装，请运行: pip install coverage"
        except Exception as e:
            coverage_data['error'] = f"覆盖率分析失败: {str(e)}"
            
        return coverage_data
    
    def get_project_complexity(self) -> Dict:
        """获取项目复杂度指标"""
        complexity = {
            'total_functions': 0,
            'total_classes': 0,
            'average_function_length': 0,
            'files_with_high_complexity': []
        }
        
        function_lengths = []
        
        for dir_name in self.source_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                for py_file in dir_path.rglob('*.py'):
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # 统计函数和类
                        functions = re.findall(r'^def\s+\w+', content, re.MULTILINE)
                        classes = re.findall(r'^class\s+\w+', content, re.MULTILINE)
                        
                        complexity['total_functions'] += len(functions)
                        complexity['total_classes'] += len(classes)
                        
                        # 估算函数长度
                        lines = content.split('\n')
                        current_function_start = None
                        indent_level = 0
                        
                        for i, line in enumerate(lines):
                            stripped = line.strip()
                            if stripped.startswith('def '):
                                if current_function_start is not None:
                                    func_length = i - current_function_start
                                    function_lengths.append(func_length)
                                current_function_start = i
                                indent_level = len(line) - len(line.lstrip())
                            elif current_function_start is not None and stripped and len(line) - len(line.lstrip()) <= indent_level and not line.startswith(' '):
                                func_length = i - current_function_start
                                function_lengths.append(func_length)
                                current_function_start = None
                        
                        # 检查高复杂度文件 (超过500行)
                        if len(lines) > 500:
                            complexity['files_with_high_complexity'].append(str(py_file.relative_to(self.project_root)))
                            
                    except Exception as e:
                        print(f"警告: 分析文件复杂度失败 {py_file}: {e}")
        
        if function_lengths:
            complexity['average_function_length'] = sum(function_lengths) / len(function_lengths)
            
        return complexity
    
    def generate_report(self) -> Dict:
        """生成完整的评估报告"""
        print("🔍 开始代码评估...")
        
        # 获取代码统计
        source_stats = self.get_source_code_stats()
        test_stats = self.get_test_code_stats()
        doc_stats = self.get_documentation_stats()
        
        # 获取测试覆盖率
        coverage_data = self.run_coverage_analysis()
        
        # 获取复杂度指标
        complexity = self.get_project_complexity()
        
        # 计算质量指标
        total_code_lines = source_stats.code_lines + test_stats.code_lines
        test_to_code_ratio = test_stats.code_lines / source_stats.code_lines if source_stats.code_lines > 0 else 0
        
        report = {
            'project_summary': {
                'source_files': source_stats.files_count,
                'test_files': test_stats.files_count,
                'documentation_files': doc_stats.files_count,
                'total_files': source_stats.files_count + test_stats.files_count + doc_stats.files_count
            },
            'code_statistics': {
                'source_code': {
                    'total_lines': source_stats.total_lines,
                    'code_lines': source_stats.code_lines,
                    'comment_lines': source_stats.comment_lines,
                    'blank_lines': source_stats.blank_lines,
                    'files': source_stats.files_count
                },
                'test_code': {
                    'total_lines': test_stats.total_lines,
                    'code_lines': test_stats.code_lines,
                    'comment_lines': test_stats.comment_lines,
                    'blank_lines': test_stats.blank_lines,
                    'files': test_stats.files_count
                },
                'documentation': {
                    'total_lines': doc_stats.total_lines,
                    'files': doc_stats.files_count
                }
            },
            'quality_metrics': {
                'test_coverage_percentage': coverage_data['coverage_percentage'],
                'test_to_code_ratio': test_to_code_ratio,
                'total_functions': complexity['total_functions'],
                'total_classes': complexity['total_classes'],
                'average_function_length': complexity['average_function_length'],
                'files_with_high_complexity': complexity['files_with_high_complexity']
            },
            'coverage_details': coverage_data,
            'complexity_analysis': complexity
        }
        
        return report


def print_report(report: Dict):
    """格式化打印报告"""
    print("\n" + "="*80)
    print("📊 QwenRag 项目代码评估报告")
    print("="*80)
    
    # 项目概览
    summary = report['project_summary']
    print(f"\n📁 项目概览:")
    print(f"   源代码文件: {summary['source_files']}")
    print(f"   测试文件: {summary['test_files']}")
    print(f"   文档文件: {summary['documentation_files']}")
    print(f"   总文件数: {summary['total_files']}")
    
    # 代码统计
    stats = report['code_statistics']
    print(f"\n📝 代码统计:")
    
    print(f"   源代码:")
    src = stats['source_code']
    print(f"     总行数: {src['total_lines']:,}")
    print(f"     代码行: {src['code_lines']:,}")
    print(f"     注释行: {src['comment_lines']:,}")
    print(f"     空白行: {src['blank_lines']:,}")
    
    print(f"   测试代码:")
    test = stats['test_code']
    print(f"     总行数: {test['total_lines']:,}")
    print(f"     代码行: {test['code_lines']:,}")
    print(f"     注释行: {test['comment_lines']:,}")
    print(f"     空白行: {test['blank_lines']:,}")
    
    # 质量指标
    metrics = report['quality_metrics']
    print(f"\n📈 质量指标:")
    print(f"   测试覆盖率: {metrics['test_coverage_percentage']:.1f}%")
    print(f"   测试/代码比: {metrics['test_to_code_ratio']:.2f}")
    print(f"   函数总数: {metrics['total_functions']}")
    print(f"   类总数: {metrics['total_classes']}")
    print(f"   平均函数长度: {metrics['average_function_length']:.1f} 行")
    
    if metrics['files_with_high_complexity']:
        print(f"   高复杂度文件: {len(metrics['files_with_high_complexity'])} 个")
        for file in metrics['files_with_high_complexity']:
            print(f"     - {file}")
    
    # 覆盖率详情
    coverage = report['coverage_details']
    if coverage.get('error'):
        print(f"\n⚠️  测试覆盖率分析: {coverage['error']}")
    else:
        print(f"\n🎯 测试覆盖率详情:")
        print(f"   覆盖率: {coverage['coverage_percentage']:.1f}%")
        print(f"   总语句数: {coverage['total_statements']:,}")
        print(f"   已覆盖语句: {coverage['covered_statements']:,}")
        print(f"   未覆盖行数: {coverage['missing_lines']:,}")
        
        # 文件级覆盖率 (只显示覆盖率较低的文件)
        low_coverage_files = {k: v for k, v in coverage.get('files_coverage', {}).items() 
                            if v['coverage'] < 80 and v['statements'] > 10}
        if low_coverage_files:
            print(f"\n   需要改进的文件 (<80% 覆盖率):")
            for file, data in sorted(low_coverage_files.items(), key=lambda x: x[1]['coverage']):
                print(f"     - {file}: {data['coverage']:.1f}% ({data['missing']} 行未覆盖)")
    
    # 建议
    print(f"\n💡 改进建议:")
    suggestions = []
    
    if metrics['test_coverage_percentage'] < 80:
        suggestions.append(f"   • 提高测试覆盖率到80%以上 (当前: {metrics['test_coverage_percentage']:.1f}%)")
    
    if metrics['test_to_code_ratio'] < 0.5:
        suggestions.append(f"   • 增加测试代码，建议测试/代码比达到0.5以上 (当前: {metrics['test_to_code_ratio']:.2f})")
    
    if metrics['average_function_length'] > 50:
        suggestions.append(f"   • 考虑重构较长的函数，建议平均长度控制在50行以内 (当前: {metrics['average_function_length']:.1f}行)")
    
    if metrics['files_with_high_complexity']:
        suggestions.append(f"   • 重构高复杂度文件，考虑拆分为更小的模块")
    
    if not suggestions:
        suggestions.append("   • 代码质量良好，继续保持！")
    
    for suggestion in suggestions:
        print(suggestion)
    
    print("\n" + "="*80)


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    evaluator = CodeEvaluator(project_root)
    
    try:
        report = evaluator.generate_report()
        print_report(report)
        
        # 保存JSON格式的详细报告
        report_file = project_root / 'eval' / 'code_evaluation_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 详细报告已保存到: {report_file}")
        
    except Exception as e:
        print(f"❌ 评估过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()