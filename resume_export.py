"""
简历 Word 导出模块
排版风格借鉴 RenderCV Classic 主题：
- 蓝色标题 RGB(0,79,144) + 右侧横线
- Source Sans 3 → Calibri（Word 通用近似）
- 页边距 0.7in，正文 10.5pt，行距紧凑
"""

import io
import re
from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml

# ─── 样式常量（借鉴 RenderCV Classic）─────────────────────
BRAND_BLUE = RGBColor(0, 79, 144)
TEXT_BLACK = RGBColor(0, 0, 0)
GRAY = RGBColor(128, 128, 128)
LIGHT_GRAY = RGBColor(180, 180, 180)

FONT_NAME = 'Calibri'
FONT_NAME_CN = '微软雅黑'

BODY_SIZE = Pt(10.5)
NAME_SIZE = Pt(26)
SECTION_TITLE_SIZE = Pt(13)
ENTRY_TITLE_SIZE = Pt(10.5)

PAGE_MARGIN = Inches(0.7)
SECTION_SPACE_BEFORE = Pt(14)
SECTION_SPACE_AFTER = Pt(6)
ENTRY_SPACE = Pt(2)
BULLET_INDENT = Cm(0.4)

# section type 排序权重
SECTION_ORDER = {
    'education': 0,
    'internship': 1,
    'project': 2,
    'competition': 3,
    'skill': 4,
    'other': 5,
}

# section type → 中文标题
SECTION_LABELS = {
    'education': '教育经历',
    'internship': '实习经历',
    'project': '项目经历',
    'competition': '竞赛经历',
    'skill': '技能证书',
    'other': '其他',
}


def _set_font(run, size=BODY_SIZE, bold=False, color=TEXT_BLACK, font_name=FONT_NAME):
    """设置 run 的字体样式"""
    run.font.size = size
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font_name
    # 中文字体回退
    r = run._element
    rPr = r.find(qn('w:rPr'))
    if rPr is None:
        rPr = parse_xml(f'<w:rPr {nsdecls("w")}></w:rPr>')
        r.insert(0, rPr)
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = parse_xml(f'<w:rFonts {nsdecls("w")}/>')
        rPr.insert(0, rFonts)
    rFonts.set(qn('w:eastAsia'), FONT_NAME_CN)


def _set_paragraph_spacing(para, before=Pt(0), after=Pt(0), line=None):
    """设置段落间距"""
    pf = para.paragraph_format
    pf.space_before = before
    pf.space_after = after
    if line is not None:
        pf.line_spacing = line


def _add_section_heading(doc, title):
    """
    添加 section 标题：蓝色文字 + 右侧横线
    借鉴 RenderCV Classic 的 with_partial_line 样式
    """
    # 用单行表格实现：左列放标题，右列放横线
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    # 去除表格边框
    tbl = table._tbl
    tblPr = tbl.find(qn('w:tblPr'))
    if tblPr is None:
        tblPr = parse_xml(f'<w:tblPr {nsdecls("w")}></w:tblPr>')
        tbl.insert(0, tblPr)
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '  <w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '  <w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '  <w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '  <w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '  <w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '  <w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '</w:tblBorders>'
    )
    tblPr.append(borders)

    # 设置表格宽度 100%
    tblW = parse_xml(f'<w:tblW {nsdecls("w")} w:type="pct" w:w="5000"/>')
    tblPr.append(tblW)

    # 去掉单元格边距
    tblCellMar = parse_xml(
        f'<w:tblCellMar {nsdecls("w")}>'
        '  <w:top w:w="0" w:type="dxa"/>'
        '  <w:left w:w="0" w:type="dxa"/>'
        '  <w:bottom w:w="0" w:type="dxa"/>'
        '  <w:right w:w="0" w:type="dxa"/>'
        '</w:tblCellMar>'
    )
    tblPr.append(tblCellMar)

    # 左列：标题文字（不换行，自适应宽度）
    cell_left = table.cell(0, 0)
    p = cell_left.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    _set_paragraph_spacing(p, before=SECTION_SPACE_BEFORE, after=Pt(0))
    run = p.add_run(title)
    _set_font(run, size=SECTION_TITLE_SIZE, bold=True, color=BRAND_BLUE)

    # 左列宽度自适应
    tc_left = cell_left._tc
    tcPr_left = tc_left.find(qn('w:tcPr'))
    if tcPr_left is None:
        tcPr_left = parse_xml(f'<w:tcPr {nsdecls("w")}></w:tcPr>')
        tc_left.insert(0, tcPr_left)
    tcW_left = parse_xml(f'<w:tcW {nsdecls("w")} w:type="auto" w:w="0"/>')
    tcPr_left.append(tcW_left)

    # 右列：底部横线
    cell_right = table.cell(0, 1)
    p2 = cell_right.paragraphs[0]
    _set_paragraph_spacing(p2, before=SECTION_SPACE_BEFORE, after=Pt(0))
    # 通过段落底部边框模拟横线
    pPr = p2._p.find(qn('w:pPr'))
    if pPr is None:
        pPr = parse_xml(f'<w:pPr {nsdecls("w")}></w:pPr>')
        p2._p.insert(0, pPr)
    pBdr = parse_xml(
        f'<w:pBdr {nsdecls("w")}>'
        f'  <w:bottom w:val="single" w:sz="4" w:space="1" w:color="004F90"/>'
        f'</w:pBdr>'
    )
    pPr.append(pBdr)

    # 右列用空 run 占位（确保有高度）
    run2 = p2.add_run(' ')
    _set_font(run2, size=SECTION_TITLE_SIZE, color=BRAND_BLUE)

    return table


def _add_entry_title(doc, title):
    """添加条目标题（如"字节跳动-产品实习"），加粗"""
    para = doc.add_paragraph()
    _set_paragraph_spacing(para, before=Pt(8), after=ENTRY_SPACE)
    run = para.add_run(title)
    _set_font(run, size=ENTRY_TITLE_SIZE, bold=True, color=TEXT_BLACK)
    return para


def _is_bullet_line(line):
    """判断是否为 bullet 行"""
    stripped = line.strip()
    return bool(re.match(r'^[•·\-*●○▪►]\s*', stripped))


def _clean_bullet(line):
    """去掉原始 bullet 符号，返回纯文本"""
    return re.sub(r'^[•·\-*●○▪►]\s*', '', line.strip())


def _add_bullet_paragraph(doc, text):
    """添加一个带 bullet 的段落"""
    para = doc.add_paragraph()
    _set_paragraph_spacing(para, before=Pt(0), after=Pt(1), line=Pt(15))
    para.paragraph_format.left_indent = BULLET_INDENT
    para.paragraph_format.first_line_indent = -BULLET_INDENT

    # bullet 符号
    run_bullet = para.add_run('•  ')
    _set_font(run_bullet, size=BODY_SIZE, color=GRAY)

    # 文本
    run_text = para.add_run(text)
    _set_font(run_text, size=BODY_SIZE, color=TEXT_BLACK)

    return para


def _add_body_paragraph(doc, text):
    """添加普通正文段落"""
    para = doc.add_paragraph()
    _set_paragraph_spacing(para, before=Pt(0), after=Pt(1), line=Pt(15))
    run = para.add_run(text)
    _set_font(run, size=BODY_SIZE, color=TEXT_BLACK)
    return para


def generate_resume_docx(resume_sections: list, user_name: str = '') -> io.BytesIO:
    """
    将 resume_sections 生成 Word 文档并返回 BytesIO 流

    参数:
        resume_sections: [{"type": "education", "title": "...", "content": "..."}, ...]
        user_name: 可选，显示在顶部的姓名

    返回:
        io.BytesIO - 可直接作为 Flask send_file 的输入
    """
    doc = Document()

    # ─── 页面设置 ────────────────────────────────
    section = doc.sections[0]
    section.top_margin = PAGE_MARGIN
    section.bottom_margin = PAGE_MARGIN
    section.left_margin = PAGE_MARGIN
    section.right_margin = PAGE_MARGIN

    # ─── 姓名标题（如果有）────────────────────────
    if user_name:
        name_para = doc.add_paragraph()
        name_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _set_paragraph_spacing(name_para, before=Pt(0), after=Pt(16))
        run = name_para.add_run(user_name)
        _set_font(run, size=NAME_SIZE, bold=True, color=BRAND_BLUE)

    # ─── 按 section type 排序 ────────────────────
    sorted_sections = sorted(
        resume_sections,
        key=lambda s: SECTION_ORDER.get(s.get('type', 'other'), 99)
    )

    # ─── 按 type 分组渲染 ────────────────────────
    current_type = None
    for sec in sorted_sections:
        sec_type = sec.get('type', 'other')
        title = sec.get('title', '')
        content = sec.get('content', '')

        # 新的 section type → 添加分类标题
        if sec_type != current_type:
            current_type = sec_type
            label = SECTION_LABELS.get(sec_type, sec_type)
            _add_section_heading(doc, label)

        # 条目标题
        if title:
            _add_entry_title(doc, title)

        # 条目内容：逐行解析
        if content:
            lines = content.split('\n')
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if _is_bullet_line(stripped):
                    _add_bullet_paragraph(doc, _clean_bullet(stripped))
                else:
                    _add_body_paragraph(doc, stripped)

    # ─── 页脚 ──────────────────────────────────
    footer = section.footer
    footer_para = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer_para.add_run('由校园人才估值平台生成')
    _set_font(run, size=Pt(8), color=LIGHT_GRAY)

    # ─── 输出 ──────────────────────────────────
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer
