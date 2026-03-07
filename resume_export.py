"""
简历 Word 导出模块
排版风格复刻 RenderCV Classic 主题：
- 蓝色 Section 标题 + 右侧蓝色横线
- 每个条目使用双栏表格：左侧标题+要点，右侧日期（右对齐）
- 紧凑行距，专业排版
"""

import io
import re
from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml

# ─── 样式常量（RenderCV Classic）─────────────────────────
BRAND_BLUE = RGBColor(0, 79, 144)
TEXT_BLACK = RGBColor(0, 0, 0)
TEXT_DARK = RGBColor(51, 51, 51)
GRAY = RGBColor(110, 110, 110)
LIGHT_GRAY = RGBColor(180, 180, 180)

FONT_BODY = 'Calibri'
FONT_CN = '微软雅黑'

SIZE_BODY = Pt(10)
SIZE_NAME = Pt(24)
SIZE_SECTION = Pt(12)
SIZE_ENTRY_TITLE = Pt(10)
SIZE_DATE = Pt(9.5)
SIZE_BULLET = Pt(10)

PAGE_MARGIN = Inches(0.7)
LINE_SPACING = Pt(14)             # 正文行距
BULLET_LINE_SPACING = Pt(13.5)    # 要点行距

# 右侧日期列宽度（RenderCV 默认 4.15cm）
DATE_COL_WIDTH = Cm(4.5)

# section type 排序 + 中文标签
SECTION_META = {
    'education':   (0, '教育经历'),
    'internship':  (1, '实习经历'),
    'project':     (2, '项目经历'),
    'competition': (3, '竞赛经历'),
    'skill':       (4, '技能证书'),
    'other':       (5, '其他'),
}

# 日期正则：匹配常见中文日期格式
_DATE_RE = re.compile(
    r'(\d{4}\s*[年./\-]\s*\d{1,2}\s*[月]?\s*[-–—~至到]+\s*'
    r'(?:\d{4}\s*[年./\-]\s*)?\d{0,2}\s*[月]?'
    r'|至今|present|今|现在'
    r'|\d{4}\s*[年./\-]\s*\d{1,2}\s*[月]?)',
    re.IGNORECASE
)

# 纯日期行判断：整行大部分内容都是日期
_DATE_LINE_RE = re.compile(
    r'^\s*\d{4}\s*[年./\-].*(?:至今|present|今|\d{1,2}\s*[月]?)\s*$',
    re.IGNORECASE
)


# ═══════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════

def _set_font(run, size=SIZE_BODY, bold=False, color=TEXT_BLACK, italic=False):
    """设置 run 字体"""
    run.font.size = size
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.italic = italic
    run.font.name = FONT_BODY
    # 中文回退
    rPr = run._element.find(qn('w:rPr'))
    if rPr is None:
        rPr = parse_xml(f'<w:rPr {nsdecls("w")}></w:rPr>')
        run._element.insert(0, rPr)
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = parse_xml(f'<w:rFonts {nsdecls("w")}/>')
        rPr.insert(0, rFonts)
    rFonts.set(qn('w:eastAsia'), FONT_CN)


def _set_spacing(para, before=Pt(0), after=Pt(0), line=LINE_SPACING):
    """设置段落间距"""
    pf = para.paragraph_format
    pf.space_before = before
    pf.space_after = after
    if line:
        pf.line_spacing = line


def _no_borders(tbl_element):
    """移除表格所有边框"""
    tblPr = tbl_element.find(qn('w:tblPr'))
    if tblPr is None:
        tblPr = parse_xml(f'<w:tblPr {nsdecls("w")}></w:tblPr>')
        tbl_element.insert(0, tblPr)
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '<w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '</w:tblBorders>'
    )
    tblPr.append(borders)
    # 100% 宽度
    tblW = parse_xml(f'<w:tblW {nsdecls("w")} w:type="pct" w:w="5000"/>')
    tblPr.append(tblW)
    # 零边距
    tblCellMar = parse_xml(
        f'<w:tblCellMar {nsdecls("w")}>'
        '<w:top w:w="0" w:type="dxa"/>'
        '<w:left w:w="0" w:type="dxa"/>'
        '<w:bottom w:w="0" w:type="dxa"/>'
        '<w:right w:w="0" w:type="dxa"/>'
        '</w:tblCellMar>'
    )
    tblPr.append(tblCellMar)
    return tblPr


def _set_cell_width(cell, width):
    """固定单元格宽度"""
    tc = cell._tc
    tcPr = tc.find(qn('w:tcPr'))
    if tcPr is None:
        tcPr = parse_xml(f'<w:tcPr {nsdecls("w")}></w:tcPr>')
        tc.insert(0, tcPr)
    tcW = parse_xml(f'<w:tcW {nsdecls("w")} w:type="dxa" w:w="{int(width.emu / 635)}"/>')
    tcPr.append(tcW)


def _set_cell_valign_top(cell):
    """单元格顶部对齐"""
    tc = cell._tc
    tcPr = tc.find(qn('w:tcPr'))
    if tcPr is None:
        tcPr = parse_xml(f'<w:tcPr {nsdecls("w")}></w:tcPr>')
        tc.insert(0, tcPr)
    vAlign = parse_xml(f'<w:vAlign {nsdecls("w")} w:val="top"/>')
    tcPr.append(vAlign)


def _add_para_bottom_border(para, color="004F90", sz="4"):
    """给段落添加底部边框线"""
    pPr = para._p.find(qn('w:pPr'))
    if pPr is None:
        pPr = parse_xml(f'<w:pPr {nsdecls("w")}></w:pPr>')
        para._p.insert(0, pPr)
    pBdr = parse_xml(
        f'<w:pBdr {nsdecls("w")}>'
        f'<w:bottom w:val="single" w:sz="{sz}" w:space="1" w:color="{color}"/>'
        f'</w:pBdr>'
    )
    pPr.append(pBdr)


# ═══════════════════════════════════════════
# 内容解析
# ═══════════════════════════════════════════

def _is_bullet(line):
    return bool(re.match(r'^[•·\-*●○▪►]\s', line.strip()))


def _clean_bullet(line):
    return re.sub(r'^[•·\-*●○▪►]\s*', '', line.strip())


def _is_date_line(line):
    """判断是否为纯日期行（整行主要是日期内容）"""
    stripped = line.strip()
    if not stripped:
        return False
    # 去掉日期部分后，剩余很少内容 → 算作日期行
    remaining = _DATE_RE.sub('', stripped).strip()
    remaining = re.sub(r'[\s\-–—~至到月年/.|]', '', remaining)
    return len(remaining) < len(stripped) * 0.4 and _DATE_RE.search(stripped) is not None


def _extract_date(line):
    """从一行中提取日期文本"""
    m = _DATE_RE.search(line)
    return m.group(0).strip() if m else ''


def _parse_entry_content(content: str):
    """
    解析条目内容，分离出：
    - date_text: 日期字符串（从第一个日期行提取）
    - header_lines: 标题/描述行（非 bullet、非日期行的前几行）
    - bullets: bullet 行列表
    - body_lines: 其他正文行
    """
    lines = content.split('\n')
    date_text = ''
    header_lines = []
    bullets = []
    body_lines = []
    found_bullet = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if _is_bullet(stripped):
            found_bullet = True
            bullets.append(_clean_bullet(stripped))
        elif _is_date_line(stripped) and not date_text:
            # 提取日期，但日期行可能还有其他内容（如角色信息）
            date_text = _extract_date(stripped)
            # 去掉日期后的残余（如果有意义的文本）
            remaining = _DATE_RE.sub('', stripped).strip()
            remaining = re.sub(r'^[\s\-–—~至到,，|]+', '', remaining).strip()
            remaining = re.sub(r'[\s\-–—~至到,，|]+$', '', remaining).strip()
            if remaining and len(remaining) > 2:
                if not found_bullet:
                    header_lines.append(remaining)
                else:
                    body_lines.append(remaining)
        elif not found_bullet and not bullets:
            header_lines.append(stripped)
        else:
            body_lines.append(stripped)

    return date_text, header_lines, bullets, body_lines


# ═══════════════════════════════════════════
# Section 标题（蓝色 + 横线）
# ═══════════════════════════════════════════

def _add_section_heading(doc, title):
    """
    RenderCV Classic 标志性样式：
    蓝色粗体标题 ──────────────────────────
    """
    table = doc.add_table(rows=1, cols=2)
    _no_borders(table._tbl)

    # 左列：标题
    cell_l = table.cell(0, 0)
    p = cell_l.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    _set_spacing(p, before=Pt(16), after=Pt(2), line=None)
    run = p.add_run(title)
    _set_font(run, size=SIZE_SECTION, bold=True, color=BRAND_BLUE)
    # 不换行
    tc_l = cell_l._tc
    tcPr_l = tc_l.find(qn('w:tcPr'))
    if tcPr_l is None:
        tcPr_l = parse_xml(f'<w:tcPr {nsdecls("w")}></w:tcPr>')
        tc_l.insert(0, tcPr_l)
    tcPr_l.append(parse_xml(f'<w:noWrap {nsdecls("w")}/>'))
    tcPr_l.append(parse_xml(f'<w:tcW {nsdecls("w")} w:type="auto" w:w="0"/>'))

    # 右列：蓝色底部横线
    cell_r = table.cell(0, 1)
    p2 = cell_r.paragraphs[0]
    _set_spacing(p2, before=Pt(16), after=Pt(2), line=None)
    _add_para_bottom_border(p2, color="004F90", sz="6")
    run2 = p2.add_run(' ')
    _set_font(run2, size=SIZE_SECTION, color=BRAND_BLUE)


# ═══════════════════════════════════════════
# 条目渲染（双栏表格布局）
# ═══════════════════════════════════════════

def _add_entry(doc, title, content):
    """
    渲染一个条目（RenderCV Classic 风格）：

    ┌──────────────────────────────────┬──────────────┐
    │ **Title**, subtitle              │   right-date │
    │ • bullet 1                       │              │
    │ • bullet 2                       │              │
    └──────────────────────────────────┴──────────────┘
    """
    date_text, header_lines, bullets, body_lines = _parse_entry_content(content)

    table = doc.add_table(rows=1, cols=2)
    _no_borders(table._tbl)

    cell_left = table.cell(0, 0)
    cell_right = table.cell(0, 1)

    _set_cell_width(cell_right, DATE_COL_WIDTH)
    _set_cell_vAlign_top(cell_left)
    _set_cell_vAlign_top(cell_right)

    # ── 左列 ──────────────────────────────
    # 条目标题行（粗体）
    p_title = cell_left.paragraphs[0]
    _set_spacing(p_title, before=Pt(7), after=Pt(1), line=LINE_SPACING)

    if title:
        run_t = p_title.add_run(title)
        _set_font(run_t, size=SIZE_ENTRY_TITLE, bold=True, color=TEXT_BLACK)

    # header_lines 作为副标题（如公司名、项目名下面的描述）
    for hl in header_lines:
        p_h = cell_left.add_paragraph()
        _set_spacing(p_h, before=Pt(0), after=Pt(1), line=LINE_SPACING)
        run_h = p_h.add_run(hl)
        _set_font(run_h, size=SIZE_BODY, color=TEXT_DARK)

    # Bullet 要点
    for b in bullets:
        p_b = cell_left.add_paragraph()
        _set_spacing(p_b, before=Pt(0), after=Pt(0.5), line=BULLET_LINE_SPACING)
        p_b.paragraph_format.left_indent = Cm(0.35)
        p_b.paragraph_format.first_line_indent = Cm(-0.35)
        run_dot = p_b.add_run('•  ')
        _set_font(run_dot, size=SIZE_BULLET, color=GRAY)
        run_text = p_b.add_run(b)
        _set_font(run_text, size=SIZE_BULLET, color=TEXT_DARK)

    # 其他正文行
    for bl in body_lines:
        p_bl = cell_left.add_paragraph()
        _set_spacing(p_bl, before=Pt(0), after=Pt(1), line=LINE_SPACING)
        run_bl = p_bl.add_run(bl)
        _set_font(run_bl, size=SIZE_BODY, color=TEXT_DARK)

    # ── 右列：日期（右对齐，顶部对齐）──────
    p_date = cell_right.paragraphs[0]
    p_date.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    _set_spacing(p_date, before=Pt(7), after=Pt(0), line=LINE_SPACING)
    if date_text:
        run_d = p_date.add_run(date_text)
        _set_font(run_d, size=SIZE_DATE, color=GRAY, italic=False)


def _set_cell_vAlign_top(cell):
    """单元格内容顶端对齐"""
    tc = cell._tc
    tcPr = tc.find(qn('w:tcPr'))
    if tcPr is None:
        tcPr = parse_xml(f'<w:tcPr {nsdecls("w")}></w:tcPr>')
        tc.insert(0, tcPr)
    # 检查是否已有 vAlign
    existing = tcPr.find(qn('w:vAlign'))
    if existing is not None:
        tcPr.remove(existing)
    vAlign = parse_xml(f'<w:vAlign {nsdecls("w")} w:val="top"/>')
    tcPr.append(vAlign)


# ═══════════════════════════════════════════
# 技能证书条目（单行格式，不用双栏）
# ═══════════════════════════════════════════

def _add_skill_entry(doc, title, content):
    """
    技能/证书条目：Label: Details（单行或多行）
    """
    lines = content.split('\n') if content else []
    all_text = [l.strip() for l in lines if l.strip()]

    para = doc.add_paragraph()
    _set_spacing(para, before=Pt(5), after=Pt(2), line=LINE_SPACING)

    if title:
        run_label = para.add_run(title + '：')
        _set_font(run_label, size=SIZE_BODY, bold=True, color=TEXT_BLACK)

    if all_text:
        run_detail = para.add_run(' '.join(all_text))
        _set_font(run_detail, size=SIZE_BODY, color=TEXT_DARK)


# ═══════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════

def generate_resume_docx(resume_sections: list, user_name: str = '') -> io.BytesIO:
    """
    将 resume_sections 生成精排 Word 文档（RenderCV Classic 风格）

    参数:
        resume_sections: [{"type": "...", "title": "...", "content": "..."}, ...]
        user_name: 可选姓名（显示在顶部）

    返回:
        io.BytesIO
    """
    doc = Document()

    # ─── 页面设置 ─────────────────────────
    sec = doc.sections[0]
    sec.top_margin = PAGE_MARGIN
    sec.bottom_margin = PAGE_MARGIN
    sec.left_margin = PAGE_MARGIN
    sec.right_margin = PAGE_MARGIN

    # ─── 姓名 ────────────────────────────
    if user_name:
        p_name = doc.add_paragraph()
        p_name.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _set_spacing(p_name, before=Pt(0), after=Pt(12), line=None)
        run = p_name.add_run(user_name)
        _set_font(run, size=SIZE_NAME, bold=True, color=BRAND_BLUE)

    # ─── 排序 ────────────────────────────
    sorted_sections = sorted(
        resume_sections,
        key=lambda s: SECTION_META.get(s.get('type', 'other'), (99, ''))[0]
    )

    # ─── 按 type 分组渲染 ────────────────
    current_type = None
    for entry in sorted_sections:
        sec_type = entry.get('type', 'other')
        title = entry.get('title', '')
        content = entry.get('content', '')

        # 新分类 → Section 标题
        if sec_type != current_type:
            current_type = sec_type
            _, label = SECTION_META.get(sec_type, (99, sec_type))
            _add_section_heading(doc, label)

        # 根据类型选择渲染方式
        if sec_type == 'skill':
            _add_skill_entry(doc, title, content)
        else:
            _add_entry(doc, title, content)

    # ─── 页脚 ─────────────────────────────
    footer = sec.footer
    fp = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run('由校园人才估值平台生成')
    _set_font(run, size=Pt(8), color=LIGHT_GRAY)

    # ─── 输出 ─────────────────────────────
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf
