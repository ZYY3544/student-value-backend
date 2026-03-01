"""
薪酬计算模块
根据职级、职能、行业、城市查询和计算薪酬（基于P50）
"""

import csv
import os
from typing import Dict, Optional, Tuple
from logger import get_module_logger
from config import config

logger = get_module_logger(__name__)


class SalaryCalculator:
    """薪酬计算器：基于HAY职级和职能查询薪酬数据"""

    def __init__(self, csv_path: str = None):
        """
        初始化薪酬计算器

        Args:
            csv_path: 薪酬数据CSV文件路径，默认使用config中配置的路径
        """
        if csv_path is None:
            csv_path = config.SALARY_CSV_PATH

        self.csv_path = csv_path
        self.salary_data = {}  # {grade: {function: {"P50_low": value, "P50_high": value}}}
        self.industry_factors = {}  # {industry: factor}
        self.city_factors = {}  # {city: factor}

        self._load_data()

    def _load_data(self):
        """
        加载CSV数据，只提取P50并计算±5%区间

        使用标记行检测不同部分，而非硬编码行号，提高对CSV格式变化的容错性
        """
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)

            if len(rows) < 3:
                raise ValueError(f"CSV文件行数不足（仅{len(rows)}行），请检查文件格式")

            # 动态查找表头行
            header_row_idx = self._find_header_row(rows)
            if header_row_idx < 0:
                logger.warning("未找到明确的表头行，使用默认索引1")
                header_row_idx = 1

            header = rows[header_row_idx]
            logger.info(f"检测到表头行: 第{header_row_idx + 1}行")

            # 检测CSV格式：三列(P25/P50/P75)还是两列(低值/高值)
            is_triple_format = any('P25' in col for col in header)

            # 解析薪酬数据：从表头行的下一行开始，直到遇到非数字开头的行
            salary_start = header_row_idx + 1
            for i, row in enumerate(rows[salary_start:], start=salary_start):
                if not row or not row[0].strip():
                    continue

                # 尝试解析职级数字
                try:
                    grade = int(row[0].strip())
                except ValueError:
                    # 遇到非数字行，可能是行业/城市系数部分
                    break

                self.salary_data[grade] = {}
                self._parse_salary_row(row, header, grade, is_triple_format)

            # 动态查找行业系数部分（查找包含"行业"的标记行）
            industry_start = self._find_section_start(rows, ['行业类型', '行业系数', '行业'])
            if industry_start >= 0:
                self._parse_factor_section(rows, industry_start, self.industry_factors, '行业')
            else:
                logger.warning("未找到行业系数部分")

            # 动态查找城市系数部分（查找包含"城市"的标记行）
            city_start = self._find_section_start(rows, ['城市类型', '城市系数', '城市'])
            if city_start >= 0:
                self._parse_factor_section(rows, city_start, self.city_factors, '城市')
            else:
                logger.warning("未找到城市系数部分")

            logger.info(f"✅ 成功加载薪酬数据：{len(self.salary_data)}个职级，{len(self.industry_factors)}个行业，{len(self.city_factors)}个城市")
            logger.info(f"   职能列表: {list(self.salary_data.get(15, {}).keys())}")

        except Exception as e:
            logger.error(f"❌ 加载薪酬数据失败: {e}")
            raise

    def _find_header_row(self, rows: list) -> int:
        """
        动态查找表头行

        Returns:
            表头行索引，未找到返回-1
        """
        for i, row in enumerate(rows[:10]):  # 只在前10行中查找
            if not row:
                continue
            row_text = ' '.join(str(cell) for cell in row)
            # 表头行通常包含这些关键词
            if any(keyword in row_text for keyword in ['P50', '职能', '薪酬']):
                return i
        return -1

    def _find_section_start(self, rows: list, keywords: list) -> int:
        """
        查找指定部分的起始行

        Args:
            rows: 所有行
            keywords: 标记关键词列表

        Returns:
            起始行索引，未找到返回-1
        """
        for i, row in enumerate(rows):
            if not row:
                continue
            first_cell = str(row[0]).strip()
            if any(keyword in first_cell for keyword in keywords):
                return i
        return -1

    def _parse_salary_row(self, row: list, header: list, grade: int, is_triple_format: bool):
        """解析单行薪酬数据，只提取P50并计算±5%区间"""
        if is_triple_format:
            # 三列格式：每个职能占3列（P25、P50、P75），只取P50
            col_idx = 1
            while col_idx + 2 < len(row):
                function_header = header[col_idx] if col_idx < len(header) else ""
                function_name = function_header.split('（')[0] if '（' in function_header else function_header.strip()

                if not function_name:
                    col_idx += 3
                    continue

                try:
                    p50 = self._parse_number(row[col_idx + 1])

                    self.salary_data[grade][function_name] = {
                        "P50_low": round(p50 * 0.95, 2),
                        "P50_high": round(p50 * 1.05, 2)
                    }
                except (IndexError, ValueError) as e:
                    logger.warning(f"解析职级{grade}的职能{function_name}时出错: {e}")

                col_idx += 3
        else:
            # 两列格式：每个职能占2列（P50低值、P50高值）
            col_idx = 1
            while col_idx + 1 < len(row):
                function_header = header[col_idx] if col_idx < len(header) else ""
                function_name = function_header.split('（')[0] if '（' in function_header else function_header.strip()

                if not function_name:
                    col_idx += 2
                    continue

                try:
                    p50_low = self._parse_number(row[col_idx])
                    p50_high = self._parse_number(row[col_idx + 1])

                    self.salary_data[grade][function_name] = {
                        "P50_low": p50_low,
                        "P50_high": p50_high
                    }
                except (IndexError, ValueError) as e:
                    logger.warning(f"解析职级{grade}的职能{function_name}时出错: {e}")

                col_idx += 2

    def _parse_factor_section(self, rows: list, start_idx: int, target_dict: dict, section_name: str):
        """
        解析系数部分（行业/城市）

        Args:
            rows: 所有行
            start_idx: 起始行索引
            target_dict: 目标字典
            section_name: 部分名称（用于日志）
        """
        for row in rows[start_idx:start_idx + 20]:  # 最多解析20行
            if not row or len(row) < 2:
                continue

            key = str(row[0]).strip()

            # 跳过标题行和空行
            if not key or key in ['行业类型', '城市类型', '行业', '城市', '系数', '']:
                continue

            # 如果遇到其他部分的标记，停止解析
            if any(marker in key for marker in ['城市', '备注', '说明']) and section_name == '行业':
                break
            if any(marker in key for marker in ['备注', '说明']) and section_name == '城市':
                break

            try:
                factor = float(str(row[1]).strip())
                target_dict[key] = factor
            except ValueError:
                continue

    def _parse_number(self, value: str) -> float:
        """
        解析数字字符串，移除空格和逗号
        注意：CSV中的数值单位为元
        """
        if not value:
            return 0.0
        # 移除空格、逗号
        cleaned = str(value).strip().replace(',', '').replace(' ', '')
        return float(cleaned)

    def get_salary_range(
        self,
        job_grade: int,
        function: str,
        industry: str,
        city: str
    ) -> Optional[Dict[str, float]]:
        """
        查询薪酬范围（P50±5%区间），应用行业和城市系数调整

        Args:
            job_grade: HAY职级（1-30）
            function: 职能名称（如"软件开发"）
            industry: 行业类型（如"互联网"）
            city: 城市类型（如"一线城市"）

        Returns:
            {"P50_low": 调整后低值, "P50_high": 调整后高值} 或 None（查询失败）
        """
        # 1. 查询基础薪酬
        if job_grade not in self.salary_data:
            logger.error(f"❌ 职级{job_grade}不在薪酬表中")
            return None

        if function not in self.salary_data[job_grade]:
            logger.error(f"❌ 职能'{function}'不在薪酬表中，可选职能: {list(self.salary_data[job_grade].keys())}")
            return None

        base_salary = self.salary_data[job_grade][function]

        # 2. 获取调整系数
        industry_factor = self.industry_factors.get(industry, 1.0)
        city_factor = self.city_factors.get(city, 1.0)

        if industry not in self.industry_factors:
            logger.warning(f"⚠️ 行业'{industry}'无系数，使用默认值1.0")
        if city not in self.city_factors:
            logger.warning(f"⚠️ 城市'{city}'无系数，使用默认值1.0")

        # 3. 计算调整后薪酬
        adjusted_salary = {
            "P50_low": round(base_salary["P50_low"] * industry_factor * city_factor, 2),
            "P50_high": round(base_salary["P50_high"] * industry_factor * city_factor, 2)
        }

        logger.info(f"✅ 薪酬查询成功: 职级{job_grade} | 职能{function} | 行业{industry}({industry_factor}) | 城市{city}({city_factor})")
        logger.info(f"   基础P50: 低值={base_salary['P50_low']:,.0f}, 高值={base_salary['P50_high']:,.0f}")
        logger.info(f"   调整后: 低值={adjusted_salary['P50_low']:,.0f}, 高值={adjusted_salary['P50_high']:,.0f}")

        return adjusted_salary

    def get_available_functions(self) -> list:
        """获取所有可用职能列表"""
        if not self.salary_data:
            return []
        # 从任意职级中获取职能列表（所有职级的职能应该一致）
        sample_grade = list(self.salary_data.keys())[0]
        return list(self.salary_data[sample_grade].keys())

    def get_available_industries(self) -> list:
        """获取所有可用行业列表"""
        return list(self.industry_factors.keys())

    def get_available_cities(self) -> list:
        """获取所有可用城市列表"""
        return list(self.city_factors.keys())


# 测试代码
if __name__ == "__main__":
    calculator = SalaryCalculator()

    # 测试查询
    result = calculator.get_salary_range(
        job_grade=15,
        function="软件开发",
        industry="互联网",
        city="一线城市"
    )

    if result:
        print(f"\n薪酬查询结果:")
        print(f"P50低值: ¥{result['P50_low']:,.0f}")
        print(f"P50高值: ¥{result['P50_high']:,.0f}")
        print(f"P50区间: {result['P50_low']/10000:.1f}万元～{result['P50_high']/10000:.1f}万元")
