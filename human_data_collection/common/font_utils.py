"""
Font Utilities Module
Provides Chinese font lookup and matplotlib configuration
"""
import warnings
import matplotlib
import matplotlib.font_manager as fm


def find_chinese_font():
    """Find available Chinese fonts in the system."""
    available_fonts = []
    chinese_fonts = [
        'Noto Sans CJK SC', 'Noto Sans CJK TC', 'Noto Sans CJK JP',
        'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
        'Arial Unicode MS', 'Source Han Sans'
    ]

    for font_name in chinese_fonts:
        try:
            if any(font_name in f.name for f in fm.fontManager.ttflist):
                available_fonts.append(font_name)
        except:
            continue

    # If no Chinese font found, use DejaVu Sans and suppress warnings
    if not available_fonts:
        available_fonts = ['DejaVu Sans']

    return available_fonts


def configure_matplotlib_fonts():
    """Configure matplotlib Chinese font support"""
    # Dynamically set available Chinese fonts
    available_chinese_fonts = find_chinese_font()
    matplotlib.rcParams['font.sans-serif'] = available_chinese_fonts + ['sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue
    matplotlib.rcParams['font.family'] = 'sans-serif'

    # Suppress font-related warnings to reduce console noise
    warnings.filterwarnings('ignore', message='.*Glyph.*missing from font.*')

    print(f"✅ Chinese fonts have been configured in matplotlib.: {available_chinese_fonts}")

    return available_chinese_fonts
