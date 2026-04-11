"""MD → DOCX 변환 스크립트

pandoc으로 DOCX를 생성하고, 테이블 테두리를 적용한 후
HWP에서 열어 HWP로 저장합니다.
사용법:
  py -3 md_to_hwp.py           # doc/설계서.md → doc/설계서.docx
  py -3 md_to_hwp.py path/to/file.md
  py -3 md_to_hwp.py --no-open # HWP 자동 실행 없이 DOCX만 생성
"""

import argparse
import copy
import os
import shutil
import subprocess
import winreg
import zipfile
from pathlib import Path

from lxml import etree

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{W_NS}}}"
_BORDER_SIDES = ("top", "left", "bottom", "right", "insideH", "insideV")


# ─────────────────────────────────────────
# pandoc 경로 탐색
# ─────────────────────────────────────────

def _find_pandoc() -> str:
    """시스템에서 pandoc 실행 파일 경로를 반환합니다."""
    p = shutil.which("pandoc")
    if p:
        return p

    def _reg_path(hive, key):
        try:
            k = winreg.OpenKey(hive, key)
            v, _ = winreg.QueryValueEx(k, "Path")
            return v
        except Exception:
            return ""

    reg_paths = (
        _reg_path(winreg.HKEY_LOCAL_MACHINE,
                  r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")
        + ";"
        + _reg_path(winreg.HKEY_CURRENT_USER, r"Environment")
    )

    for segment in reg_paths.split(";"):
        segment = os.path.expandvars(segment.strip())
        if not segment:
            continue
        candidate = os.path.join(segment, "pandoc.exe")
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "pandoc을 찾을 수 없습니다.\n"
        "https://pandoc.org/installing.html 에서 설치 후 재시도하세요."
    )


# ─────────────────────────────────────────
# HWP 실행 파일 탐색
# ─────────────────────────────────────────

def _find_hwp() -> str | None:
    """HWP 실행 파일 경로를 반환합니다. 없으면 None."""
    candidates = [
        r"C:\Program Files (x86)\HNC\Office 2018\HOffice100\Bin\Hwp.exe",
        r"C:\Program Files (x86)\HNC\Office 2020\HOffice110\Bin\Hwp.exe",
        r"C:\Program Files\HNC\Office 2022\HOffice120\Bin\Hwp.exe",
        r"C:\Program Files\HNC\Office NEO\HOffice90\Bin\Hwp.exe",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


# ─────────────────────────────────────────
# DOCX 테이블 테두리 후처리 (lxml + zipfile)
# python-docx 방식은 로드/저장 시 XML 손상 위험이 있으므로
# ZIP을 직접 열어 document.xml만 수정합니다.
# ─────────────────────────────────────────

def _make_tc_borders() -> etree._Element:
    """tcBorders XML 요소를 생성합니다."""
    borders = etree.Element(f"{W}tcBorders")
    for side in _BORDER_SIDES:
        el = etree.SubElement(borders, f"{W}{side}")
        el.set(f"{W}val", "single")
        el.set(f"{W}sz", "4")
        el.set(f"{W}space", "0")
        el.set(f"{W}color", "000000")
    return borders


def _patch_document_xml(xml_bytes: bytes) -> bytes:
    """document.xml의 모든 테이블에 테두리를 추가하고 width=0 테이블을 수정합니다."""
    root = etree.fromstring(xml_bytes)
    borders_template = _make_tc_borders()

    for tbl in root.iter(f"{W}tbl"):
        # width=0 테이블을 페이지 폭 100%로 수정 (HWP/Word에서 안 보이는 문제 해결)
        tblPr = tbl.find(f"{W}tblPr")
        if tblPr is not None:
            tblW = tblPr.find(f"{W}tblW")
            if tblW is not None and tblW.get(f"{W}w") == "0":
                tblW.set(f"{W}type", "pct")
                tblW.set(f"{W}w", "5000")

        # 셀 테두리 적용
        for tc in tbl.iter(f"{W}tc"):
            tcPr = tc.find(f"{W}tcPr")
            if tcPr is None:
                tcPr = etree.Element(f"{W}tcPr")
                tc.insert(0, tcPr)

            for old in tcPr.findall(f"{W}tcBorders"):
                tcPr.remove(old)

            tcPr.insert(0, copy.deepcopy(borders_template))

    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def _apply_table_borders(docx_path: Path) -> None:
    """DOCX ZIP 내 document.xml을 직접 수정해 테이블 테두리를 추가합니다."""
    tmp_path = docx_path.with_suffix(".tmp.docx")

    with zipfile.ZipFile(docx_path, "r") as zin, \
         zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                data = _patch_document_xml(data)
            zout.writestr(item, data)

    docx_path.unlink()
    tmp_path.rename(docx_path)


# ─────────────────────────────────────────
# MD → DOCX (pandoc)
# ─────────────────────────────────────────

def md_to_docx(md_path: Path, docx_path: Path, pandoc: str) -> None:
    """pandoc으로 마크다운 파일을 DOCX로 변환합니다."""
    cmd = [
        pandoc,
        str(md_path),
        "-o", str(docx_path),
        "--standalone",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"pandoc 변환 실패:\n{result.stderr}")
    print(f"pandoc 변환 완료: {docx_path}")

    _apply_table_borders(docx_path)
    print(f"테이블 테두리 적용 완료: {docx_path}")


# ─────────────────────────────────────────
# HWP에서 DOCX 열기
# ─────────────────────────────────────────

def open_in_hwp(docx_path: Path, hwp_exe: str) -> None:
    """DOCX 파일을 HWP에서 엽니다."""
    abs_docx = str(docx_path.resolve())
    print(f"HWP에서 열기: {abs_docx}")
    subprocess.Popen([hwp_exe, abs_docx])
    print("HWP가 실행되었습니다.")
    print("파일 > 다른 이름으로 저장 > HWP 형식으로 저장하세요.")


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MD → DOCX 변환 (pandoc) + HWP에서 열기")
    parser.add_argument(
        "md_file",
        nargs="?",
        default=None,
        help="변환할 .md 파일 경로 (기본값: doc/설계서.md)",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="변환 후 HWP 자동 실행 안 함 (DOCX만 생성)",
    )
    args = parser.parse_args()

    base = Path(__file__).parent

    if args.md_file:
        md_path = Path(args.md_file)
    else:
        md_path = base / "설계서.md"

    docx_path = md_path.with_suffix(".docx")

    pandoc = _find_pandoc()
    print(f"pandoc: {pandoc}")

    print("=== MD → DOCX ===")
    md_to_docx(md_path, docx_path, pandoc)

    if args.no_open:
        print(f"\n완료: {docx_path}")
        print("한글(HWP)에서 위 파일을 열고 HWP 형식으로 저장하세요.")
        return

    print("\n=== HWP에서 열기 ===")
    hwp_exe = _find_hwp()
    if hwp_exe:
        open_in_hwp(docx_path, hwp_exe)
    else:
        print("HWP 실행 파일을 찾을 수 없습니다.")
        print(f"직접 HWP에서 열어주세요: {docx_path}")

    print(f"\n생성된 파일: {docx_path}")


if __name__ == "__main__":
    main()
