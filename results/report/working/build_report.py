from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
from docx.opc.constants import RELATIONSHIP_TYPE as RT

ROOT = Path(r"D:/PythonProject/llm-compute-allocation-modeling-main")
REF = Path(r"C:/Users/paulb/Desktop/F26105740044.docx")
OUT = ROOT / "results/report/算力约束下大语言模型资源配置实验报告.docx"
Q1 = ROOT / "results/q1"
Q2 = ROOT / "results/q2"
Q3 = ROOT / "results/q3"
Q4 = ROOT / "results/q4"


def set_run_font(run, name="宋体", size=10.5, bold=False, italic=False, color="000000"):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Times New Roman" if name == "宋体" else name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Times New Roman" if name == "宋体" else name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)


def set_cell_shading(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tcPr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_border(cell, **kwargs):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = tcPr.first_child_found_in("w:tcBorders")
    if tcBorders is None:
        tcBorders = OxmlElement("w:tcBorders")
        tcPr.append(tcBorders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        if edge in kwargs:
            tag = "w:%s" % edge
            element = tcBorders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                tcBorders.append(element)
            for key in ["val", "sz", "space", "color"]:
                if key in kwargs[edge]:
                    element.set(qn("w:%s" % key), str(kwargs[edge][key]))


def set_cell_text(cell, text, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, size=9.1, color="000000"):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.05
    r = p.add_run(str(text))
    set_run_font(r, size=size, bold=bold, color=color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    set_cell_border(cell, top={"val": "single", "sz": 4, "color": "B7B7B7"}, bottom={"val": "single", "sz": 4, "color": "B7B7B7"}, left={"val": "single", "sz": 4, "color": "D9D9D9"}, right={"val": "single", "sz": 4, "color": "D9D9D9"})


def add_table(doc, headers, rows, widths=None, font_size=8.9):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    # Repeat the header when Word splits a long table across pages.
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    table.rows[0]._tr.get_or_add_trPr().append(tbl_header)
    for i, h in enumerate(headers):
        set_cell_text(table.rows[0].cells[i], h, bold=True, size=font_size, color="FFFFFF")
        set_cell_shading(table.rows[0].cells[i], "4F5B66")
    for ridx, row in enumerate(rows):
        cells = table.add_row().cells
        cant_split = OxmlElement("w:cantSplit")
        cells[0]._tc.getparent().get_or_add_trPr().append(cant_split)
        for i, val in enumerate(row):
            set_cell_text(cells[i], val, size=font_size, align=WD_ALIGN_PARAGRAPH.LEFT if isinstance(val, str) and len(val) > 22 else WD_ALIGN_PARAGRAPH.CENTER)
            if ridx % 2 == 1:
                set_cell_shading(cells[i], "F2F4F5")
    if widths:
        for row in table.rows:
            for i, width in enumerate(widths):
                row.cells[i].width = Cm(width)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    return table


def add_caption(doc, text, kind="图"):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    set_run_font(r, size=9.5, bold=False)
    return p


def add_body(doc, text, bold_lead=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Cm(0.74)
    p.paragraph_format.line_spacing = 1.28
    p.paragraph_format.space_after = Pt(4)
    if bold_lead and text.startswith(bold_lead):
        r = p.add_run(bold_lead)
        set_run_font(r, bold=True)
        r = p.add_run(text[len(bold_lead):])
        set_run_font(r)
    else:
        r = p.add_run(text)
        set_run_font(r)
    return p


def add_bullet(doc, text, level=0):
    p = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    p.paragraph_format.left_indent = Cm(0.74 + level * 0.5)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.18
    r = p.add_run(text)
    set_run_font(r, size=10)
    return p


def add_numbered(doc, text):
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.left_indent = Cm(0.74)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    set_run_font(r, size=10)
    return p


def add_figure(doc, path, caption, note=None, width_cm=14.65, max_height_cm=7.2):
    path = Path(path)
    if not path.exists():
        add_body(doc, f"图文件缺失：{path.name}。该结果未能嵌入报告。")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    run = p.add_run()
    # Use the native image ratio, limiting the physical height for page stability.
    from PIL import Image
    with Image.open(path) as im:
        ratio = im.height / max(im.width, 1)
    w = width_cm
    h = w * ratio
    if h > max_height_cm:
        h = max_height_cm
        w = h / ratio
    run.add_picture(str(path), width=Cm(w), height=Cm(h))
    try:
        inline = run._r.xpath('.//wp:inline')[0]
        docPr = inline.xpath('./wp:docPr')[0]
        docPr.set('descr', caption)
        docPr.set('title', caption)
    except Exception:
        pass
    add_caption(doc, caption)
    if note:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.first_line_indent = Cm(0.74)
        p.paragraph_format.line_spacing = 1.18
        p.paragraph_format.space_after = Pt(5)
        r = p.add_run(note)
        set_run_font(r, size=9.5, italic=True, color="555555")


def m_run(text, italic=False, bold=False):
    r = OxmlElement("m:r")
    rPr = OxmlElement("m:rPr")
    sty = OxmlElement("m:sty")
    sty.set(qn("m:val"), "i" if italic else "p")
    rPr.append(sty)
    r.append(rPr)
    t = OxmlElement("m:t")
    t.text = str(text)
    r.append(t)
    return r


def m_group(*children):
    e = OxmlElement("m:e")
    for c in children:
        e.append(c if hasattr(c, "tag") else m_run(c))
    return e


def m_sub(base, sub):
    e = OxmlElement("m:sSub")
    e.append(OxmlElement("m:sSubPr"))
    e.append(m_group(base if hasattr(base, "tag") else m_run(base, italic=True)))
    e.append(m_group(sub if hasattr(sub, "tag") else m_run(sub)))
    return e


def m_sup(base, sup):
    e = OxmlElement("m:sSup")
    e.append(OxmlElement("m:sSupPr"))
    e.append(m_group(base if hasattr(base, "tag") else m_run(base, italic=True)))
    e.append(m_group(sup if hasattr(sup, "tag") else m_run(sup)))
    return e


def m_subsup(base, sub, sup):
    e = OxmlElement("m:sSubSup")
    e.append(OxmlElement("m:sSubSupPr"))
    e.append(m_group(base if hasattr(base, "tag") else m_run(base, italic=True)))
    e.append(m_group(sub if hasattr(sub, "tag") else m_run(sub)))
    e.append(m_group(sup if hasattr(sup, "tag") else m_run(sup)))
    return e


def m_frac(num, den):
    e = OxmlElement("m:f")
    e.append(OxmlElement("m:fPr"))
    e.append(m_group(num if hasattr(num, "tag") else m_run(num)))
    e.append(m_group(den if hasattr(den, "tag") else m_run(den)))
    return e


def m_nary(sub, sup, body, symbol="∑"):
    e = OxmlElement("m:nary")
    pr = OxmlElement("m:naryPr")
    chr_ = OxmlElement("m:chr")
    chr_.set(qn("m:val"), symbol)
    pr.append(chr_)
    lim = OxmlElement("m:limLoc")
    lim.set(qn("m:val"), "undOvr")
    pr.append(lim)
    e.append(pr)
    e.append(m_group(sub if hasattr(sub, "tag") else m_run(sub)))
    e.append(m_group(sup if hasattr(sup, "tag") else m_run(sup)))
    e.append(m_group(body if hasattr(body, "tag") else m_run(body)))
    return e


def add_equation(doc, label, pieces):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.keep_with_next = True
    omath_para = OxmlElement("m:oMathPara")
    omath = OxmlElement("m:oMath")
    for part in pieces:
        omath.append(part if hasattr(part, "tag") else m_run(part))
    omath_para.append(omath)
    p._p.append(omath_para)
    if label:
        r = p.add_run(f"  （{label}）")
        set_run_font(r, size=9.5)
    return p


def add_page_field(paragraph, instr="PAGE"):
    run = paragraph.add_run()
    fldChar1 = OxmlElement("w:fldChar")
    fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText")
    instrText.set(qn("xml:space"), "preserve")
    instrText.text = instr
    fldChar2 = OxmlElement("w:fldChar")
    fldChar2.set(qn("w:fldCharType"), "end")
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    set_run_font(run, size=9)


def add_toc(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = 'TOC \\o "1-3" \\h \\z \\u'
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    txt = OxmlElement("w:t")
    txt.text = "目录将在打开文档时更新。"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, sep, txt, end])
    set_run_font(run, size=10.5)


def clear_document(doc):
    body = doc._element.body
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)
    for section in doc.sections:
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin = Cm(3.175)
        section.right_margin = Cm(3.175)
        section.header_distance = Cm(1.501)
        section.footer_distance = Cm(1.75)
        section.different_first_page_header_footer = True
        # The retained template has several linked header/footer parts. Clear
        # every block element (including paragraphs hidden in tables/text boxes)
        # so only the new page field remains and Word cannot display duplicate
        # template page numbers.
        for part in (section.header, section.first_page_header, section.footer, section.first_page_footer):
            root = part._element
            for child in list(root):
                if child.tag != qn("w:sectPr"):
                    root.remove(child)
            root.append(OxmlElement("w:p"))
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_page_field(footer)


def setup_styles(doc):
    styles = doc.styles
    for name, base in [("Heading 1", "Normal"), ("Heading 2", "Normal"), ("Heading 3", "Normal"), ("Caption", "Normal"), ("List Bullet", "Normal"), ("List Bullet 2", "List Bullet"), ("List Number", "Normal")]:
        try:
            styles[name]
        except KeyError:
            styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
            if base in styles:
                styles[name].base_style = styles[base]
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(10.5)
    normal.paragraph_format.line_spacing = 1.28
    normal.paragraph_format.space_after = Pt(4)
    normal.paragraph_format.first_line_indent = Cm(0.74)
    for name, size, bold, align, before, after in [("Title", 22, True, WD_ALIGN_PARAGRAPH.CENTER, 0, 16), ("Heading 1", 14, True, WD_ALIGN_PARAGRAPH.CENTER, 12, 8), ("Heading 2", 11.5, True, WD_ALIGN_PARAGRAPH.LEFT, 8, 4), ("Heading 3", 10.5, True, WD_ALIGN_PARAGRAPH.LEFT, 5, 3), ("Caption", 9.5, False, WD_ALIGN_PARAGRAPH.CENTER, 2, 5)]:
        st = styles[name]
        st.font.name = "黑体" if name.startswith("Heading") else "宋体"
        st._element.rPr.rFonts.set(qn("w:eastAsia"), st.font.name)
        st.font.size = Pt(size)
        st.font.bold = bold
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.paragraph_format.alignment = align
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.keep_with_next = True
    if "List Bullet" in styles:
        styles["List Bullet"].font.size = Pt(10)
        styles["List Bullet"]._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "宋体")


def heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.add_run(text)
    for r in p.runs:
        set_run_font(r, name="黑体", size={1: 14, 2: 11.5, 3: 10.5}[level], bold=True)
    return p


def fmt(x, n=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return str(x)


def csv_rows(path, cols=None, n=None):
    df = pd.read_csv(path)
    if cols:
        df = df[cols]
    if n:
        df = df.head(n)
    return df


def cleanup_unused_assets(docx_path):
    # The reference's custom XML is not used for the report. Remove both the
    # parts and their package declarations; leaving a dangling relationship
    # makes python-docx/Word reject the otherwise valid package.
    from xml.etree import ElementTree as ET

    rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    ct_ns = "http://schemas.openxmlformats.org/package/2006/content-types"
    ET.register_namespace("", rel_ns)
    ET.register_namespace("", ct_ns)
    temp = docx_path.with_suffix(".clean.docx")
    with zipfile.ZipFile(docx_path, "r") as zin, zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in zin.namelist():
            if name.startswith("customXml/"):
                continue
            data = zin.read(name)
            if name.endswith(".rels"):
                root = ET.fromstring(data)
                for rel in list(root):
                    if rel.attrib.get("Type", "").endswith("/customXml") or rel.attrib.get("Target", "").replace("\\", "/").find("customXml/") >= 0:
                        root.remove(rel)
                data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
            elif name == "[Content_Types].xml":
                root = ET.fromstring(data)
                for child in list(root):
                    if child.attrib.get("PartName", "").replace("\\", "/").startswith("/customXml/"):
                        root.remove(child)
                data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
            zout.writestr(name, data)
    os.replace(temp, docx_path)


def detach_reference_images(doc):
    parts = [doc.part]
    for section in doc.sections:
        parts.extend([section.header.part, section.first_page_header.part, section.footer.part, section.first_page_footer.part])
    for part in parts:
        for rid, rel in list(part.rels.items()):
            if rel.reltype == RT.IMAGE:
                del part.rels[rid]


def main():
    doc = Document(str(REF))
    clear_document(doc)
    detach_reference_images(doc)
    setup_styles(doc)

    # Cover
    for _ in range(5):
        doc.add_paragraph()
    p = doc.add_paragraph(style="Title")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("算力约束下大语言模型资源配置建模")
    for r in p.runs:
        set_run_font(r, name="黑体", size=22, bold=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(18)
    r = p.add_run("问题一至问题四实验报告")
    set_run_font(r, name="黑体", size=15, bold=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("基于项目现有数据、模型与结果图表")
    set_run_font(r, size=11.5)
    for _ in range(7):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("实验报告")
    set_run_font(r, name="黑体", size=14, bold=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("2026年10月")
    set_run_font(r, size=11)
    doc.add_page_break()

    heading(doc, "摘要", 1)
    add_body(doc, "本文围绕算力约束下大语言模型能力提升的资源配置问题，对项目中问题一至问题四的完整实验流程进行整理。问题一从 22 个质量指标出发，比较统一方向与按七个质量域设置方向的质量评价，随后采用两段式语义冲突扣减，并在配比数据上比较岭回归、共享系数的 M1、逐损失域系数的 M1 和非线性的 M2。问题二在 B1 经典标度律上固定经验基线 E，比较 additive、interaction 和新的质量幂函数三种质量响应，并逐一比较加入 H* 常数修正与不加入的六个模型。问题三使用问题二选出的模型，在资源预算、上下文长度、质量成本和配比单纯形约束下，以粒子群算法进行优化，并以 SLSQP 作对照。问题四保持项目原方法，对开源模型能力前沿、规模与技术分解、Loss–能力桥接和六项任务能力进行分析。")
    add_body(doc, "报告特别区分了拟合改进与机制识别。当前 B 数据逐行没有领域配比观测，因此 H* 在问题二中对所有样本相同，lambda·H* 等效于可调截距；开放配比 p 的第三问优化依赖于 A 数据中 H(p) 形状可以迁移到 B 的额外假设。Q 和 p 可以作为控制变量分别进入代理模型，但这不意味着二者在真实数据中统计独立，也不允许同时把 Q 定义为 p 的加权函数后又把二者自由优化。所有半合成数据、跨尺度估计和超出拟合范围的资源解均按情景或诊断处理，不表述为已完成的真实训练验证。")
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("关键词：")
    set_run_font(r, bold=True)
    r = p.add_run("数据质量；领域配比；标度律；粒子群优化；外推；大语言模型")
    set_run_font(r)
    doc.add_page_break()

    heading(doc, "目录", 1)
    add_toc(doc)
    doc.add_page_break()

    heading(doc, "1 研究问题与总体路线", 1)
    add_body(doc, "项目研究的共同目标是在给定训练算力的条件下，解释模型损失如何随模型规模 N、训练数据量 D、数据质量 Q 和领域配比 p 变化，并据此给出资源分配方案。四个问题的输入来源和可识别对象不同：问题一的 A 数据同时含有文本质量指标和配比–损失实验；问题二的 B 数据主要用于标度律拟合，其中 B6–B8 是基于真实数据校准的半合成补充；问题三把前两问得到的响应面作为代理目标；问题四分析独立的模型能力与开源演化数据。")
    add_table(doc, ["问题", "主要输入", "核心输出", "验证口径"], [
        ["问题一", "A1–A3质量样本；A4–A15配比–损失表", "七域质量Q、语义修正、M1/M2回归", "训练/测试、嵌套交叉验证、跨尺度诊断"],
        ["问题二", "B1–B8；Q1导出的M2与H*", "经典与广义标度律、六项拟合比较", "按(N,D)分组五折、N/D留出、B6→B7、B8迁移"],
        ["问题三", "问题二选优公式、预算与成本假设", "N、D、Q、p最优配置", "PSO与SLSQP、固定/自由配比、条件前后"],
        ["问题四", "C1、C3、C4、C6、C8", "能力前沿、分解、桥接和情景预测", "回归拟合、分组误差、情景扰动"],
    ], widths=[1.8, 5.0, 4.2, 3.65])
    add_body(doc, "整体流程先用质量指标得到域内质量 Q，再用配比实验建立 p 对损失的代理响应 H(p)。问题二中不重复读取 A 原始文件，而是读取问题一导出的 M2 参数和最优配比；问题三再将选优标度律和 H(p) 组合为约束优化目标。这样做保持了实验来源的独立性，但也把跨来源迁移作为必须明确写出的模型假设。")
    add_equation(doc, "总体代理结构", [m_run("L", italic=True), " = ", m_run("L", italic=True), m_sub("scale", ""), " + ", m_run("lambda", italic=True), m_run("H", italic=True), "(", m_run("p", italic=True), ")"])

    heading(doc, "2 数据、符号与模型假设", 1)
    heading(doc, "2.1 数据范围与变量定义", 2)
    add_body(doc, "A 质量数据包含 A1–A3 共 272,505 条记录，其中 arxiv 17,523 条、github 203,752 条，第三个 A3 数据按项目清单合并。七个质量域的样本数为 arxiv 18,942、book 171、c4 10,000、commoncrawl 9,640、github 213,752、stackexchange 10,000、wikipedia 10,000。github 占比约 78.44%，因此全体样本直接求均值会被单一域主导。质量指标固定为项目当前的 22 个字段，不加入截图中的额外 QuRating 子维度；若某语义冲突字段不存在，则该冲突对跳过。")
    indicators = [
        ("1–3", "dsir_books；dsir_wiki；dsir_math", "标量", "语料分布相似度"),
        ("4–7", "rps_doc_word_count；rps_doc_num_sentences；rps_doc_unigram_entropy；rps_doc_frac_unique_words", "标量", "长度与词汇统计"),
        ("8–10", "rps_doc_frac_no_alph_words；rps_doc_frac_chars_top_2gram；rps_doc_frac_chars_top_3gram", "标量", "字符与非字母比例"),
        ("11–14", "rps_lines_uppercase_letter_fraction；rps_lines_ending_with_terminal_punctution_mark；rps_lines_numerical_chars_fraction；rps_doc_mean_word_length", "标量", "行级和词级统计"),
        ("15–18", "fineweb_edu；fluency_en；modernbert_cleanliness；modernbert_readability", "列表均值", "教育性与可读性"),
        ("19–22", "modernbert_reasoning；modernbert_professionalism；qurater；ad_en", "列表均值", "推理、专业、质量与广告信号"),
    ]
    add_table(doc, ["指标序号", "字段名", "输入形态", "解释"], indicators, widths=[1.5, 8.25, 2.1, 2.8], font_size=8.5)
    add_body(doc, "记 N、D、Q、p 分别为模型参数规模、训练数据量、质量分数和 17 维领域配比；L 为交叉熵损失，E 为固定的 B1 经验基线。质量域数 7、配方域数 17、损失域数 13 并不相同，当前实现没有伪造七域到十七域的质量映射。A 的七域质量用于质量校准，A 的 17 域配方用于 M2 的 H(p)，B 的 Q 仍使用其原始 Q_score。")

    heading(doc, "2.2 需要承担的假设", 2)
    add_numbered(doc, "质量方向假设：在每个质量域内，指标的正、负或中性方向由 direction_mapping.csv 指定。方向设置可以因域而异，但归一化分位点、熵权和 CRITIC 权重仍在全体合并样本上计算；因此“分领域方向化”不是对七个域分别重新标准化，也不是给每个域重新赋一套权重。")
    add_numbered(doc, "Q 与 p 的控制假设：在当前代理模型中，Q 表示在给定配比下通过筛选、清洗或域内处理可改变的平均质量，p 表示覆盖与任务适配。二者可在优化器中作为两个控制变量输入，但这只是可分离的工程近似，不是统计独立性结论。若另行规定 q_k 固定且 Q=Σp_k q_k，则 Q 已由 p 决定，不能继续把 Q、p 当作两个自由变量。")
    add_numbered(doc, "H* 迁移假设：H(p) 由 A 中 13 个损失域的 M2 预测构造，H* 是最优配比相对于均匀 17 域配比的预测损失差。问题二把每条 B 记录的 H 都设为同一个 H*，等价于假设 B 的每个样本都处于 A 的最优配比情景，而不是声称观测到 B 的实际 p。")
    add_numbered(doc, "可识别性假设：因为 B 行的 H* 相同，E+lambda H* 是一个有效截距，lambda 与 E 的组合不能从 B 单独识别出真实配比效应。将 lambda 解释为配比迁移强度，需要 B 中有逐行配比或至少有多种配比条件的观测。")
    add_numbered(doc, "外推假设：将 A 的 H(p) 形状迁移到 B，再迁移到问题三的预算场景，要求损失域定义、训练协议和质量尺度在来源之间可比。B6–B8 为半合成补充集，10B、70B 的 A 记录是估计数据，因此相关结论只能称为模型诊断或情景结果。")
    add_numbered(doc, "优化解释假设：成本函数、边界和上下文长度是代理设定，不是一次真实训练实验。PSO 与 SLSQP 的差异反映在给定参数化和目标函数上的数值求解表现，不自动转化为训练收益或因果结论。")

    heading(doc, "3 问题一 质量建模与配比回归", 1)
    heading(doc, "3.1 指标预处理与领域方向", 2)
    add_body(doc, "对每个指标先用全体记录的 1% 和 99% 分位点裁剪，再做 min–max 归一化得到 u。正向指标取 P=u，负向指标取 N=1−u，中性指标取 0.5。方向矩阵由七个质量域分别指定，域之间允许不同方向。例如字符重复度、过多非字母字符、异常大写比例等通常按负向使用；教育性、流畅性和可读性等按正向使用。方向有争议的字段不会被解释为已经得到外部盲评证明，结果仅表示在当前规则下的敏感性。")
    add_figure(doc, Q1 / "01_direction_matrix.png", "图3-1 七个质量域的指标方向矩阵", "矩阵显示的是方向规则，而不是域内重新训练的权重。统一归一化保证域间可比，域特定方向负责表达同一指标在不同文本类型中的含义差异。")
    add_figure(doc, Q1 / "02_domain_quality_comparison.png", "图3-2 原始方向与分领域方向的域质量比较", "分领域方向使 github、stackexchange、wikipedia 的分数明显上升，而 arxiv、book 下降；这主要来自方向规则改变，不应直接等同于真实质量提升。")
    add_figure(doc, Q1 / "03_indicator_weights_comparison.png", "图3-3 指标权重比较", "熵权与 modified CRITIC 各占一半。全体样本被 github 主导，权重变化与方向变化共同作用于最终排序。")
    add_figure(doc, Q1 / "04_sample_score_comparison.png", "图3-4 抽样记录的质量分数比较", "抽样层面展示了方向修改对个体质量分数的影响，说明域内规则会改变样本排序而不仅是改变域均值。")
    add_body(doc, "组合权重取熵权与 modified CRITIC 的平均。CRITIC 部分使用标准差乘以与其他指标的 Spearman 非相关度。质量主分数是加权和，TOPSIS 作为补充排序。全体方向与分领域方向的域均值如下，数值来自 main_domain_quality.csv 的 Q0 列。")
    domain_rows = [[d, n, fmt(qo, 3), fmt(qn, 3), fmt(delta, 3), ro, rn] for d, n, qo, qn, delta, ro, rn in [
        ("arxiv", "18,942", .463395, .463395, 0.000, 2, 7), ("book", "171", .583161, .583161, 0.000, 1, 2), ("c4", "10,000", .429242, .482359, .053, 3, 5), ("commoncrawl", "9,640", .407975, .499486, .092, 4, 4), ("github", "213,752", .385891, .598841, .213, 5, 1), ("stackexchange", "10,000", .339026, .474896, .136, 7, 6), ("wikipedia", "10,000", .364512, .517928, .153, 6, 3)]]
    add_table(doc, ["质量域", "n", "原始Q", "分领域Q", "变化", "原始排名", "新排名"], domain_rows, widths=[3.1, 1.8, 2.1, 2.2, 2.0, 2.1, 2.1])
    add_body(doc, "原始排序为 book、arxiv、c4、commoncrawl、github、wikipedia、stackexchange；分领域方向后为 github、book、wikipedia、commoncrawl、c4、stackexchange、arxiv。ALL 样本新旧质量的 Spearman 相关约 0.150，top-10 重合约 0.086，说明方向规则是实质性改变，而非微小数值修补。该结果支持开展敏感性分析，但不支持在没有外部质量标签时断言新排序必然更正确。")
    add_figure(doc, Q1 / "05_conflict_comparison.png", "图3-5 冲突样本的旧方法与语义修正比较", "旧方法按每条冲突记录删除最低三项、最高三项后聚合；新方法保留样本并施加连续惩罚。两者改变的是冲突处理，不是 22 指标集合。")
    add_figure(doc, Q1 / "06_score_change_decomposition.png", "图3-6 质量分数变化分解", "分解显示方向变化、权重变化和冲突修正的相对贡献。固定顺序分解是解释性分解，并非因果贡献。")
    add_figure(doc, Q1 / "07_ranking_stability.png", "图3-7 排名稳定性", "排名相关和 top-k 重合度用于判断方法敏感性；低重合意味着规则选择必须在报告中透明披露。")

    heading(doc, "3.2 两段式语义冲突修正", 2)
    add_body(doc, "原始冲突对共 11 对，但当前 22 个字段中只有 6 对的字段同时存在，缺失字段不以聚合值代替，而是直接跳过。对一条记录，若一对指标的归一化方向值一高一低，且最高值不小于 0.8、最低值不大于 0.2，则记为强冲突。冲突强度按差值超过 0.6 的部分线性计算，并乘以合并数据的有效冲突率。")
    add_equation(doc, "冲突强度", [m_run("S", italic=True), " = ", m_run("C", italic=True), "(", m_frac("|u_j-u_k|-0.6", "0.4"), ")"])
    add_equation(doc, "语义扣减", [m_run("T", italic=True), m_sub("i", ""), " = ", m_frac("1", "6"), m_nary("j,k", "6", "C", "∑")])
    add_body(doc, "代码保留了连续熵扣减与语义扣减的接口：Q_after_entropy=Q0−lambda_E R_i，再减去 lambda_S T_i。由于当前项目没有逐样本熵罚分输入，实际主结果只计算 Q0−lambda_S T_i，并明确标为 Q_semantic_only；熵权赋权不能冒充熵罚分。默认演示参数为 alpha=0.5、lambda_S=0.1。该限制是方法边界，不应在报告中写成两段熵–语义修正已经被数据验证。")
    add_figure(doc, Q1 / "08_semantic_quality_comparison.png", "图3-8 语义冲突修正前后的域质量", "book 和 arxiv 因冲突率较高而扣减更多，c4 与 commoncrawl 的变化较小；扣减是规则输出，不等价于发现这些域真实质量更低。")
    add_figure(doc, Q1 / "09_semantic_pair_rates.png", "图3-9 有效语义冲突对的冲突率", "fluency–readability、字符二元/三元比例、unique–entropy、terminal punctuation–fluency 等六对参与计算。缺失的 QuRating 子字段按接口要求跳过。")
    add_figure(doc, Q1 / "10_semantic_deduction_distribution.png", "图3-10 语义扣减分布", "扣减分布展示长尾与域差异，提示统一 lambda 只是一种可解释的演示设置。")
    add_figure(doc, Q1 / "11_semantic_parameter_sensitivity.png", "图3-11 语义扣减参数敏感性", "alpha 和 lambda 的变化会改变排序与域均值。没有外部标签时，参数敏感性比单一参数点更有信息。")
    add_body(doc, "最终主质量表中的 Q_new 为分领域方向加 Q_semantic_only。七个域的最终值为 arxiv 0.446712、book 0.555441、c4 0.481824、commoncrawl 0.497347、github 0.597022、stackexchange 0.474698、wikipedia 0.516852。arxiv、book、github 的平均扣减分别约 0.01697、0.02898、0.00218；扣减率高并不能证明域本身低质，因为它也可能表示两个指标的定义混合了长度或文本类型效应。")

    heading(doc, "3.3 问题一第三小问的配比–损失回归", 2)
    add_body(doc, "17 个配方域、13 个损失域和 7 个质量域承担不同作用。A4/A5 的 1M、512 条配方用于训练，A6/A7 的 1M、256 条用于测试；A8/A9 为 60M，A10/A11 为 1B，A12–A15 为 10B 和 70B 的估计配方。配比数值按原文件读取，未把约 0.996–1.003 的舍入值强行重新归一化。")
    add_equation(doc, "M1共享系数", [m_sub("L", "ij"), " = ", m_sub("c", "j"), " − ", m_nary("k=1", "17", m_group(m_run("w", italic=True), m_run(" log(1+p", italic=True), m_sub("k", ""), m_run(")")), "∑")])
    add_equation(doc, "M2主模型", [m_sub("L", "ij"), " = ", m_sub("c", "j"), " + ", m_nary("k=1", "17", m_group(m_run("b", italic=True), m_sub("kj", ""), m_run(" log(1+p", italic=True), m_sub("k", ""), m_run("/tau)")), "∑")])
    add_body(doc, "M1 共享配比系数，M2 对每个损失域使用 b_kj 并通过 tau 控制低配比域的敏感度。岭回归包括原始 alpha=1 与在训练折内调参的 alpha。五折按配方组划分，嵌套交叉验证用于避免把测试信息带入超参数选择；跨尺度指标只是迁移诊断，10B 与 70B 的损失为估计值。")
    add_table(doc, ["模型", "1M测试R²", "RMSE", "Spearman", "解释"], [
        ["Ridge 原始", "0.5782", "0.4740", "0.8141", "线性基线"],
        ["Ridge 调参", "0.6185", "0.4539", "0.8319", "alpha 在训练内选择"],
        ["M1 共享", "0.0476", "0.6841", "0.2804", "共用 w，表达力不足"],
        ["M1 独立", "0.7003", "0.4060", "0.8623", "逐损失域 w_kj"],
        ["M2", "0.8832", "0.2224", "0.9460", "主模型，tau=0.0003"],
    ], widths=[3.1, 2.1, 2.2, 2.4, 4.85])
    add_body(doc, "M2 相对原始岭回归的 1M 测试 RMSE 下降约 53.08%，与调参岭回归相比下降约 50.99%；2000 次配对 bootstrap 的 RMSE 差相对 Ridge_original 为 −0.2516，95% 区间 [−0.2637,−0.2399]。跨尺度去均值诊断的 R² 在 1M 为 0.8851、60M 为 0.6952、1B 为 −2.7933、10B 估计为 −4.3369、70B 估计为 −7.7437，说明 1M 内拟合好并不等于跨规模外推好。")
    add_figure(doc, Q1 / "12_mixture_model_comparison.png", "图3-12 配比回归模型比较", "M2 在 1M 内测试上表现最好，但跨尺度诊断下降，因而后续 H(p) 被当作可迁移假设而不是已证明规律。")
    add_figure(doc, Q1 / "13_mixture_1m_predictions.png", "图3-13 1M 配比–损失预测", "预测图展示 M2 对 1M 测试损失的拟合能力。其优势不能自动外推到没有配比观测的 B 数据。")
    add_figure(doc, Q1 / "14_mixture_tau_profile.png", "图3-14 M2 的 tau 敏感性", "tau 在最优点附近控制小配比域的放大程度；最优 tau 小于最小正配比，约 45.13% 的配比为零，需谨慎解释为物理常数。")

    heading(doc, "4 问题二 广义标度律", 1)
    heading(doc, "4.1 经典标度律与数据迁移", 2)
    add_body(doc, "经典模型使用 B1 的 1,176 条记录拟合 E+A N^(−a)+B D^(−b)，并固定 E=1.6897975629。这个 E 是 B1 数据上的经验基线，不是跨语料、跨模型都相同的理论不可约损失。B1 内拟合 R²=0.9999998、RMSE=0.0001465；迁移到 Cerebras 族时原始 R²=−5.0728，去均值后的诊断 R²=0.3848；跨族收敛点为 0.6045（去偏移 0.8300），文献数据为 0.7306（去偏移 0.7335）。去偏移只是诊断，不改变原始目标，也不能掩盖来源间截距差异。")
    add_figure(doc, Q2 / "00_classical_baseline.png", "图4-1 经典标度律及跨来源诊断", "B1 内拟合几乎完美，但跨来源原始 R² 受截距偏移影响明显；去偏移 R² 仅作为结构诊断。")
    add_body(doc, "B6 与 B7 合并去除 360 条完全重复记录后得到 450 条唯一数据；按 (N,D) 整组做五折，另设最大 N、最大 D 留出，并用 B6 训练、B7 新 Q 条件做迁移诊断。B8 的 1,704 条保留原始 Q，仅用于迁移诊断。所有六个模型使用相同数据、折划分、初值数量、解析 Jacobian 和有界最小二乘设置。")

    heading(doc, "4.2 六项广义模型与 H*", 2)
    add_equation(doc, "经典项", [m_run("L", italic=True), " = ", m_run("E", italic=True), " + ", m_run("A", italic=True), m_sup("N", "−a"), " + ", m_run("B", italic=True), m_sup("D", "−b")])
    add_equation(doc, "additive", [m_run("L", italic=True), " = ", m_run("E", italic=True), " + ", m_run("A", italic=True), m_sup("N", "−a"), " + ", m_run("B", italic=True), m_sup("D", "−b"), " + ", m_run("C", italic=True), m_sup("(1−Q)", "g")])
    add_equation(doc, "interaction", [m_run("L", italic=True), " = ", m_run("E", italic=True), " + ", m_run("A", italic=True), m_sup("N", "−a"), " + ", m_run("B", italic=True), m_sup("D", "−b"), " + ", m_run("C", italic=True), m_sup("(1−Q)", "g"), m_sup("D", "−d")])
    add_equation(doc, "quality_power", [m_run("L", italic=True), " = ", m_run("E", italic=True), " + ", m_run("A", italic=True), m_sup("N", "−a"), " + ", m_run("B", italic=True), m_sup("D", "−b"), " + ", m_run("eta", italic=True), "(1−", m_sup("Q", "kappa"), ")"])
    add_equation(doc, "加入配比常数", [m_run("L", italic=True), m_sub("H", ""), " = ", m_run("L", italic=True), " + ", m_run("lambda", italic=True), m_run("H", italic=True), "*"])
    add_body(doc, "三种质量项各计算不含 H* 和含 lambda H* 两个版本，共六项。H(p)=F(p)−F(p_ref)，其中 F 是 13 个损失域的 M2 预测损失等权平均，p_ref 是均匀 17 域配比；最优配比下 H*=−0.0896732441，F*=4.3884912，F_ref=4.4781645。H* 不是熵，也不是绝对损失。由于 H* 对 B 的每行相同，加入 H* 之后 E_eff=E+lambda H*，因此 lambda 只在 E 被固定时具有数值意义，仍不能独立识别 p 的作用。")
    add_figure(doc, Q2 / "01_validation_comparison.png", "图4-2 六项模型的验证比较", "OOF 结果显示加入 H* 的 additive 模型最优，但其提升首先应解释为截距校正。")
    add_figure(doc, Q2 / "02_oof_predictions.png", "图4-3 六项模型的分组 OOF 预测", "OOF 预测按 (N,D) 分组留出，避免同一资源组的近重复记录同时进入训练和验证。")
    add_figure(doc, Q2 / "03_quality_response.png", "图4-4 三种质量响应曲线", "quality_power 的 kappa 接近 1，additive 的 g 也接近 1，因此在本数据范围内三种质量项近似线性，性能差异很小。")
    perf_rows = [
        ["additive", "0.969047", "0.059930", "0.046310", "0.891289", "0.965740"],
        ["interaction", "0.969160", "0.059821", "0.046210", "0.894091", "0.970196"],
        ["quality_power", "0.969003", "0.059973", "0.046345", "0.890608", "0.965593"],
        ["additive + H*", "0.970582", "0.058426", "0.045350", "0.891876", "0.971673"],
        ["interaction + H*", "0.969696", "0.059299", "0.045960", "0.892380", "0.973158"],
        ["quality_power + H*", "0.970568", "0.058440", "0.045387", "0.891371", "0.971630"],
    ]
    add_table(doc, ["形式", "OOF R²", "OOF RMSE", "OOF MAE", "N留出R²", "D留出R²"], perf_rows, widths=[4.0, 2.1, 2.3, 2.2, 2.1, 2.0])
    add_body(doc, "按 OOF RMSE 选择 additive+H* 作为问题三主模型。相对于不加 H*，additive、interaction、quality_power 的 OOF RMSE 分别下降 2.510%、0.872%、2.557%。quality_power+H* 比 additive+H* 的 RMSE 高 0.024%，没有显著优势；它相对 interaction+H* 的 OOF RMSE 低 1.450%，但在 N/D 留出上并不一致。B8 原始 Q 迁移 R² 全部约为 −2.74 至 −2.83，不能宣称跨来源外推成功。")
    add_body(doc, "六项模型的直接结论是：H* 对验证有小幅帮助，但更像固定 E 下的截距校正；质量响应形式的选择影响较小，当前数据不足以证明新 quality_power 全面优于 additive 或 interaction。要识别真实配比机制，需要 B 中包含不同 p 的成组实验，或把 Q、p、H(p) 通过共同的可观测映射联合校准。")

    heading(doc, "5 问题三 算力约束下资源优化", 1)
    heading(doc, "5.1 优化目标、成本与隐含条件", 2)
    add_body(doc, "问题三使用问题二选出的 additive+H* 响应面作为主模型，同时保留 interaction+H* 作公式对照。N、D 用十亿为单位，质量 Q 的范围为 [0.4,1]，N_B 和 D_B 的范围分别为 [0.005,10^5]、[0.2,10^5]。预算取 10^19、10^22、10^24 FLOPs；主上下文长度为 2048 和 4096，另对 8192、32768、131072 作敏感性分析。")
    add_equation(doc, "训练与注意力成本", [m_run("C", italic=True), " = ", m_sup("10", "18"), m_run("N", italic=True), m_run("D", italic=True), "(", "6+", m_run("eta", italic=True), m_sub("L", "ctx"), ")"])
    add_equation(doc, "质量成本", [m_sub("C", "quality"), " = ", m_sup("10", "9"), m_run("D", italic=True), "[", m_run("f", italic=True), "(", m_run("Q", italic=True), ")−", m_run("f", italic=True), "(0.4)]", m_sub("+", "")])
    add_body(doc, "训练成本为 10^18 N_B D_B (6+eta L_ctx)，质量成本为 10^9 D_B [f(Q)−f(0.4)]_+，注意力成本与训练成本同量纲。三种 f(Q) 为 exp、power、log。由于 D 在固定 N、Q 时单调增加损失的边界内可直接取预算允许的最大值，代码先精确消去 D，再优化 N、Q 和可选的 p；另用参考扫描与加密检查复核。")
    add_body(doc, "隐含条件包括：Q 可以在固定 p 下独立变化；p 不产生额外成本；lambda>0 且 H(p) 只有 A 的形状可以迁移到当前场景；17 个配方域供应无限且配比只受单纯形约束；连续的 N、D 可以由某个实际架构实现。任何一个条件不成立，‘自由优化 p 没有收益’的结论都可能改变。")
    add_figure(doc, Q3 / "01_model_comparison.png", "图5-1 additive 与 interaction 的模型比较", "两种响应面在同一成本和边界下比较。interaction 在代表性低预算点倾向更多参数、较少数据，但这只是当前拟合参数的结果。")
    add_figure(doc, Q3 / "02_solver_gap.png", "图5-2 PSO 与 SLSQP 的目标差距", "两种求解器都以参考最优值计算 gap；PSO 和 SLSQP 的差异主要表现为数值求解过程，而非模型公式差异。")
    add_figure(doc, Q3 / "03_solver_runtime.png", "图5-3 两种求解器运行时间", "固定配比低维问题中 SLSQP 更快；自由配比问题中 PSO 的批量向量计算更快。速度优势不等于评估次数更少。")
    add_figure(doc, Q3 / "04_recipe_comparison.png", "图5-4 固定配比与自由配比的资源配置比较", "在当前可分离目标中，开放 p 后解回到 A 的 p*，因此配置几乎重合。")
    add_figure(doc, Q3 / "05_context_cost.png", "图5-5 上下文长度对成本和配置的影响", "上下文长度越长，注意力成本份额越高，最优方案会减少 N、D 并增加注意力支出份额。")
    add_figure(doc, Q3 / "06_pso_convergence.png", "图5-6 粒子群算法收敛曲线", "PSO 使用 128 粒子、800 轮、惯性权重 0.9→0.4、c1=c2=1.7，并重复五次。")
    add_figure(doc, Q3 / "07_resource_model_comparison.png", "图5-7 三种质量成本函数的资源配置", "exp、power、log 质量成本改变预算在训练、质量和注意力之间的分配，但不能解释成真实质量成本已被实验测量。")
    add_figure(doc, Q3 / "08_constraints_before_after.png", "图5-8 引入隐含条件前后的情景比较", "‘前’采用理论上下文相等点 30000，‘后’采用主情景 2048/4096。上下文变化改变了任务本身，因此损失差不能称作同一任务上的纯改进。")
    add_table(doc, ["预算", "公式", "N(B)", "D(B)", "Q", "Loss", "训练份额", "质量份额", "注意力份额"], [
        ["10^19", "additive", "0.2704", "4.7021", "0.5354", "3.2922", "0.7629", "0.1329", "0.1042"],
        ["10^19", "interaction", "0.2806", "4.3283", "0.5692", "3.2936", "0.7287", "0.1718", "0.0995"],
        ["10^22", "additive", "8.7248", "155.3561", "1.0000", "2.1003", "0.8133", "0.0757", "0.1110"],
        ["10^22", "interaction", "9.9267", "137.8095", "1.0000", "2.1030", "0.8208", "0.0671", "0.1121"],
        ["10^24", "additive", "85.5234", "1700.4690", "1.0000", "1.8140", "0.8726", "0.0083", "0.1191"],
        ["10^24", "interaction", "105.3997", "1381.9524", "1.0000", "1.8312", "0.8739", "0.0067", "0.1193"],
    ], widths=[1.4, 2.0, 1.55, 1.8, 1.3, 1.6, 1.65, 1.65, 1.7], font_size=7.9)

    heading(doc, "5.2 配比自由优化的解释", 2)
    add_equation(doc, "可分离情形", [m_run("min", italic=True), " L(", m_run("N", italic=True), ",", m_run("D", italic=True), ",", m_run("Q", italic=True), ",", m_run("p", italic=True), ") = ", m_run("min", italic=True), " L", m_sub("scale", ""), "(", m_run("N", italic=True), ",", m_run("D", italic=True), ",", m_run("Q", italic=True), ") + ", m_run("lambda", italic=True), m_run("min", italic=True), " H(", m_run("p", italic=True), ")"])
    add_body(doc, "在这个公式中，p 只出现在 lambda H(p)，成本与 Q 不含 p，且 lambda 为正。因此联合最优化可拆成先求 p*=argmin H(p)，再求 N、D、Q；自由 p 的最优值必然回到固定 p*。实际结果中 PSO 的最大 free–fixed Loss 差为 1.33×10^−15，SLSQP 为 6.23×10^−11，均属于数值误差。这个结果不是‘配比不重要’，而是当前公式没有让 p 与 Q、成本或任务权重发生耦合。")
    add_table(doc, ["配比模式", "算法", "中位时间(s)", "中位评估次数", "命中率", "结论"], [
        ["fixed", "PSO", "0.1057", "102,612", "1.000", "解精度达到参考"],
        ["fixed", "SLSQP", "0.0603", "469", "1.000", "低维时更快"],
        ["free", "PSO", "0.1909", "102,612", "1.000", "自由配比时更快"],
        ["free", "SLSQP", "1.5009", "19,561.5", "1.000", "有限差分成本高"],
    ], widths=[2.2, 2.1, 3.0, 3.0, 2.0, 2.35])
    add_body(doc, "PSO 的优势是自由配比高维搜索中的批量计算和不依赖梯度；固定配比的低维问题中 SLSQP 更快，且本实验没有证据表明 PSO 更精确或具有全局最优保证。两者使用相同的可行域参数化，不能把所有改善归因于算法。预算 10^19 的 D 低于 B 拟合下界，10^24 的 N、D 高于 B 支持范围，表中配置属于代理模型外推。")

    heading(doc, "6 问题四 模型能力演化与桥接", 1)
    heading(doc, "6.1 能力前沿与规模技术分解", 2)
    add_body(doc, "问题四保持现有方法：C1 清理开源模型排行榜，C3 使用时间序列，C4 提取 epoch 与参数信息，C6 建立 Loss–能力桥接，C8 聚合六类任务。前沿模型通过 0.9 分位数回归建模：ln S = 2.480669 + 0.364315 ln N + 0.089045(year−2022)，样本数 2,493。C1 按 Submission Date 筛选，license 用正则匹配 apache、mit、bsd、cc-by、llama、gemma、open 等关键词；自定义许可并不等价于统一 OSI 许可。代码未按 pretrained/chat 分层，也没有将 C4 的主要参数、日期、openweights 和算力字段全部纳入因果控制，因此本节是统计描述，不是因果估计。")
    add_figure(doc, Q4 / "q4_frontier.png", "图6-1 能力前沿与规模关系", "能力前沿由分位数回归给出，体现高分位模型随规模和年份的统计趋势。")
    add_figure(doc, Q4 / "q4_decomposition.png", "图6-2 规模与非规模技术分解", "分解定义为 ΔlnS=规模项+非规模项。2021–2024 的规模份额约 38.40%，分母很小，不能作常规贡献率解释。")
    add_body(doc, "分段分解结果为：2019–2020 总增长 2.3026，其中规模 1.7339、非规模 0.5687；2020–2021 总增长 0，规模 −1.2288、非规模 1.2288；2021–2024 总增长 0.0243，规模 0.9343、非规模 −0.9100；2024–2025 总增长 −0.0816，规模 −0.3158、非规模 0.2342。2024–2025 的观测前沿从 51.2313 降到 47.2169，说明年度前沿与累积最大值口径不同。")

    heading(doc, "6.2 Loss–能力桥接与情景预测", 2)
    add_equation(doc, "桥接模型", [m_run("logit", italic=True), "(", m_run("A", italic=True), ") = 0.851812 − 3.304270", m_run("log", italic=True), "(", m_run("Loss", italic=True), ")"])
    add_body(doc, "桥接回归使用 75 条记录，在 logit 能力尺度上的 R²=0.2759。高组 7 条记录的预测误差均值为 7.9625，标准差 2.6199；中组 68 条记录均值为 −3.1257，标准差 10.2610。两组来自不同验证口径，不能把误差均值直接拼接为统一的模型精度。")
    add_figure(doc, Q4 / "q4_bridge.png", "图6-3 Loss 与能力桥接", "桥接关系的解释力有限，误差分组差异较大，不能将问题三的代理损失直接宣称为真实 benchmark 提升。")
    add_figure(doc, Q4 / "q4_c8_tasks.png", "图6-4 C8 六项任务能力分布", "六项任务的均值分别为 BBH 0.4749、IFEval 0.4017、MUSR 0.3996、MMLU 0.3199、GPQA 0.2980、MATH 0.1132；各任务缺失模型数不同，均值不应粗略称为同一种准确率。")
    add_body(doc, "C4 参数增长模型给出 g_N=1.251125。以 2025 年基准、固定扰动 σ=0.12 的情景模拟预测，12 个月 base 中位数为 71.7060、slowdown 为 57.1054；24 个月 base 为 123.9170、slowdown 为 78.5591。24 个月 base 超过能力上限 100，属于模型失效情景，报告保留原始数值并标明不能裁剪为 100；区间是固定扰动下的 P10–P90 情景范围，不是重抽样置信区间。")

    heading(doc, "7 综合结论、限制与可检验的后续工作", 1)
    add_body(doc, "第一问说明质量建模的主要敏感源是方向规则、冲突处理和样本不平衡。分领域方向让域间排序发生明显变化，语义扣减可连续处理有效冲突，但当前没有熵罚分输入，也没有外部人工质量标签。第三小问的 M2 在 1M 配比测试上明显优于岭回归与共享系数 M1，但跨尺度诊断下降，不能将 H(p) 当作跨来源已验证定律。")
    add_body(doc, "第二问中，additive+H* 的 OOF RMSE 最低，H* 带来约 0.87%–2.56% 的误差下降，但 H* 对 B 为常数，因此其机制解释是截距修正而非配比效应识别。quality_power 的 kappa 接近 1，与 additive 的近线性质量惩罚相近，没有显示稳定优势。第三问在可分离代理中重新开放 p 不带来实质收益，这是公式结构的数学结果；要研究 p 与 Q 的真实联动，需要有同一 N、D 下多种 p 和多种域内质量处理的联合实验。")
    add_body(doc, "问题三的 PSO 在自由配比高维问题上运行更快，固定配比低维问题中 SLSQP 更快；不能据此说 PSO 普遍优于最小二乘规划。问题四的能力前沿和桥接可作为外部背景与量纲检查，但当前没有将 pretrained/chat、C4 算力和数据量做完整分层控制，也不能把 Q3 代理最优点闭合为真实 benchmark 提升。")
    add_body(doc, "后续实验应至少包含三类设计：第一，在每个质量域固定配比并改变域内质量处理，检验 Q 是否可在不同 p 下独立调节；第二，在相同 N、D、Q 下改变 p，并记录真实训练损失，检验 H(p) 是否跨规模迁移；第三，补齐 B 中的逐行 p 或建立经过校准的七域到十七域映射，再比较联合模型、交互项模型和可分离模型。只有在这些数据条件下，lambda 才能从截距参数升级为可解释的配比效应参数。")

    heading(doc, "附录 A 主要参数与配比", 1)
    add_body(doc, "问题二选中的 additive+H* 全量参数为 A=0.529124、a=0.283218、B=1.328565、b=0.299575、C=0.362013、g=0.990520、lambda=1.884377、E_eff=1.520819。H*=−0.089673，E 为 1.689798。问题三主流程读取该选优模型，不重新拟合标度律参数。")
    pstar = [0.065941,0.059206,0.010120,0.082819,0.060945,0.129520,0.069749,0.014526,0.127709,0.010386,0.027128,0.105455,0.102644,0.0,0.040485,0.048675,0.044692]
    domains = ["arxiv","freelaw","nih_exporter","pubmed_central","wikipedia_en","dm_mathematics","github","philpapers","stackexchange","enron_emails","gutenberg_pg_19","pile_cc","ubuntu_irc","europarl","hackernews","pubmed_abstracts","uspto_backgrounds"]
    add_table(doc, ["配方域", "p*", "配方域", "p*"], [[domains[i], fmt(pstar[i], 6), domains[i+9] if i+9 < len(domains) else "", fmt(pstar[i+9], 6) if i+9 < len(domains) else ""] for i in range(9)], widths=[4.2, 2.4, 4.2, 2.4], font_size=8.5)

    heading(doc, "附录 B 结果文件与复现入口", 1)
    add_table(doc, ["脚本", "实验内容", "主要输出"], [
        ["q1_quality.py", "Q1.1–1.2 质量方向与冲突", "main_domain_quality.csv；14张图"],
        ["q1_mixture.py", "Q1.3 M1/M2/岭回归", "mixture_models.json；3张图"],
        ["q2_baseline.py", "经典标度律与迁移", "classical_baseline.json；00图"],
        ["q2_scaling.py", "六项广义标度律", "model_performance.csv；01–03图"],
        ["q3_optimization.py", "PSO、SLSQP与情景约束", "main_optimal_configurations.csv；8张图"],
        ["q4_evolution.py", "能力前沿、桥接与任务", "q4_results.json；4张图"],
        ["validate_project.py", "全流程审计", "project_validation.json"],
    ], widths=[3.5, 6.0, 5.15], font_size=8.5)
    add_body(doc, "完整 CSV、JSON、分析 markdown 和 PNG 保留在 results/q1 至 results/q4。运行入口为 python src/run_all.py；独立运行前须满足 README 中的依赖顺序。报告生成辅助脚本位于 results/report/working，仅用于本文档构建与渲染 QA。")

    heading(doc, "参考资料", 1)
    refs = [
        "项目题目文件：《算力约束下提升大语言模型能力的资源配置建模》。",
        "项目数据说明文件及 real_attachments 中的 A、B、C 数据附件。",
        "Kaplan, J. et al. Scaling Laws for Neural Language Models. arXiv:2001.08361, 2020.",
        "Hoffmann, J. et al. Training Compute-Optimal Large Language Models. arXiv:2203.15556, 2022.",
        "Biderman, S. et al. Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. ICML, 2023.",
        "RegMix 数据混合回归实验及其项目内 A 数据导出结果。",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.74)
        p.paragraph_format.first_line_indent = Cm(-0.74)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(ref)
        set_run_font(r, size=10)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT))
    cleanup_unused_assets(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
