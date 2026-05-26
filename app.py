import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import core
from modules import bai10 as bai10_new, bai11 as bai11_new, bai12 as bai12_new

st.set_page_config(
    page_title="AIDEOM-VN — 12 bài thực hành",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root { --pink:#ec4899; --purple:#7c3aed; --cyan:#22d3ee; --card:#111827; --line:#243047; }
.block-container { padding-top: 2rem; max-width: 1280px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0b1020 0%,#111827 100%); border-right: 1px solid #263249; }
[data-testid="stSidebar"] * { color: #e5e7eb; }
h1, h2, h3 { letter-spacing: .2px; }
.badge { display:inline-block; padding:6px 12px; border-radius: 999px; background: linear-gradient(90deg,var(--pink),var(--purple)); color: white; font-weight:700; font-size: 12px; margin-right:8px; }
.smallbadge { display:inline-block; padding:4px 9px; border-radius: 999px; background:#1f2937; color:#d1d5db; font-size:12px; border:1px solid #374151; margin-right:6px; }
.card { background: rgba(17,24,39,.92); border: 1px solid #25304a; border-radius: 20px; padding: 18px 20px; box-shadow: 0 18px 40px rgba(0,0,0,.22); margin: 10px 0; }
.card h3 { margin-top:0; }
.metric-card { background: linear-gradient(135deg, rgba(236,72,153,.16), rgba(124,58,237,.12)); border: 1px solid rgba(236,72,153,.35); border-radius: 20px; padding: 16px 18px; min-height: 116px; }
.metric-label { color: #aab3c6; font-size: 13px; }
.metric-value { color: white; font-size: 28px; font-weight: 800; margin-top: 6px; }
.metric-help { color: #94a3b8; font-size: 12px; margin-top: 6px; }
hr { border-color: #263249; }
.stButton > button { border-radius:999px; border:0; background: linear-gradient(90deg,#ec4899,#7c3aed); color:white; font-weight:700; padding:.65rem 1.1rem; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { background:#111827; border-radius:999px; padding:8px 14px; border:1px solid #263249; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"


def hero(title, level, subtitle):
    st.markdown(f"<span class='badge'>{level}</span><span class='smallbadge'>Streamlit + Python</span><span class='smallbadge'>AIDEOM-VN</span>", unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def metric_cards(items):
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{value}</div>
                <div class='metric-help'>{help_text}</div>
            </div>
            """, unsafe_allow_html=True)


def df_show(df, use_container_width=True, hide_index=True):
    st.dataframe(df, use_container_width=use_container_width, hide_index=hide_index)


def fmt_df(df: pd.DataFrame, precision: int = 3):
    return df.style.format(precision=precision)


def line_fig(df, x, ys, title, labels=None):
    fig = px.line(df, x=x, y=ys, markers=True, template=PLOTLY_TEMPLATE, title=title, labels=labels or {})
    fig.update_layout(legend_title_text="", margin=dict(l=20,r=20,t=50,b=20), height=430)
    return fig


def bar_fig(df, x, y, title, color=None, orientation="v"):
    fig = px.bar(df, x=x, y=y, color=color, template=PLOTLY_TEMPLATE, title=title, orientation=orientation)
    fig.update_layout(legend_title_text="", margin=dict(l=20,r=20,t=50,b=20), height=440)
    return fig


@st.cache_data(show_spinner=False)
def cached_bai7(n, seed):
    return core.bai7_pareto(n_samples=n, seed=seed)


@st.cache_data(show_spinner=False)
def cached_q(episodes, seed):
    return core.train_q_learning(episodes=episodes, seed=seed)


# Sidebar
st.sidebar.markdown("### 🇻🇳 AIDEOM-VN")
st.sidebar.caption("Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỉ nguyên AI")
menu = [
    "🏠 Trang chủ",
    "📈 Bài 1 — Cobb-Douglas + AI",
    "💰 Bài 2 — LP ngân sách số",
    "📊 Bài 3 — Priority 10 ngành",
    "🧭 Bài 4 — LP ngành-vùng",
    "🧩 Bài 5 — MIP chọn dự án",
    "🏆 Bài 6 — TOPSIS 6 vùng",
    "🌐 Bài 7 — Pareto NSGA-II",
    "⏳ Bài 8 — Tối ưu động 2026-2035",
    "👷 Bài 9 — Lao động & AI",
    "🎲 Bài 10 — Stochastic SP",
    "🤖 Bài 11 — Q-learning RL",
    "🖥️ Bài 12 — Dashboard tích hợp",
]
page = st.sidebar.radio("Chọn bài", menu, index=0)
page_id = menu.index(page)  # FIX: dùng index chính xác, tránh Bài 10/11/12 bị nhận nhầm là Bài 1
st.sidebar.markdown("---")
st.sidebar.markdown("**Dữ liệu:** Macro, Sectors, Regions 2020–2025")
st.sidebar.markdown("**Stack:** numpy · pandas · scipy · plotly · streamlit")

# -----------------------------------------------------------------------------
# Home
# -----------------------------------------------------------------------------
if page_id == 0:
    hero("AIDEOM-VN — Dashboard 12 bài thực hành", "TỔNG HỢP", "Giao diện này thực hiện đầy đủ các yêu cầu lập trình chính của bộ bài tập: LP, MIP, TOPSIS, Pareto, động, stochastic programming và Q-learning.")
    metric_cards([
        ("Số bài", "12", "Từ dễ đến khó"),
        ("Module tích hợp", "6", "M1–M6 theo đồ án"),
        ("Kịch bản", "5", "S1–S5 đến 2030"),
        ("Môi trường", "Local", "VS Code + Streamlit"),
    ])
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Bản đồ bài tập")
    df_map = pd.DataFrame({
        "Cấp độ": ["Dễ", "Dễ", "Dễ", "Trung bình", "Trung bình", "Trung bình", "Khá khó", "Khá khó", "Khá khó", "Khó", "Khó", "Khó"],
        "Bài": [f"Bài {i}" for i in range(1,13)],
        "Nội dung": [
            "Cobb-Douglas mở rộng", "LP ngân sách", "Priority ngành", "LP vùng-hạng mục", "MIP chọn dự án", "TOPSIS vùng",
            "Pareto đa mục tiêu", "Tối ưu động", "Tác động lao động", "Quy hoạch ngẫu nhiên", "Q-learning", "Dashboard tích hợp"
        ],
        "Kết quả chính": [
            "TFP, MAPE, GDP 2030", "Phân bổ x1-x4, shadow price", "Top ngành ưu tiên", "Ma trận 6×4, chi phí công bằng", "Tập dự án tối ưu", "Top vùng AI",
            "Tập Pareto, nghiệm thỏa hiệp", "Quỹ đạo K,D,AI,H,Y,C", "NetJob và Sankey", "SP, VSS, EVPI", "π*, learning curve", "So sánh S1-S5"
        ]
    })
    df_show(df_map)
    st.markdown("</div>", unsafe_allow_html=True)
    st.info("Chọn từng bài ở thanh bên trái để xem đề bài, mô hình, kết quả tính toán và biểu đồ.")

# -----------------------------------------------------------------------------
# Bai 1
# -----------------------------------------------------------------------------
elif page_id == 1:
    hero("Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng với AI và số hóa", "DỄ", "Ước lượng TFP A_t, dự báo GDP, phân rã tăng trưởng và mô phỏng GDP 2030.")
    with st.expander("Đề bài tóm tắt", expanded=True):
        st.markdown("""
        Mô hình: **Yₜ = Aₜ · Kₜ^α · Lₜ^β · Dₜ^γ · AIₜ^δ · Hₜ^θ**, với α+β+γ+δ+θ=1.  
        Yêu cầu: tính Aₜ, lấy A trung bình để dự báo Ŷₜ và MAPE, phân rã tăng trưởng 2020–2025, mô phỏng GDP 2030.
        """)
    c1,c2,c3,c4,c5 = st.columns(5)
    alpha = c1.number_input("α K", value=0.33, min_value=0.0, max_value=1.0, step=0.01)
    beta = c2.number_input("β L", value=0.42, min_value=0.0, max_value=1.0, step=0.01)
    gamma = c3.number_input("γ D", value=0.10, min_value=0.0, max_value=1.0, step=0.01)
    delta = c4.number_input("δ AI", value=0.08, min_value=0.0, max_value=1.0, step=0.01)
    theta = c5.number_input("θ H", value=0.07, min_value=0.0, max_value=1.0, step=0.01)
    out = core.bai1_cobb_douglas(alpha,beta,gamma,delta,theta)
    metric_cards([
        ("A trung bình", f"{out['A_bar']:.4f}", "TFP calibrated"),
        ("MAPE", f"{out['MAPE']:.2f}%", "Sai số dự báo trong mẫu"),
        ("GDP 2030", f"{out['Y2030']:,.1f}", "nghìn tỷ VND"),
        ("Tổng hệ số", f"{alpha+beta+gamma+delta+theta:.2f}", "lợi suất theo quy mô"),
    ])
    tab1, tab2, tab3 = st.tabs(["TFP & dự báo", "Phân rã tăng trưởng", "Kịch bản 2030"])
    with tab1:
        st.plotly_chart(line_fig(out["table"], "year", ["A_TFP"], "TFP Aₜ theo năm"), use_container_width=True)
        st.plotly_chart(line_fig(out["table"], "year", ["GDP_trillion_VND", "Y_hat"], "GDP thực tế và dự báo Ŷₜ"), use_container_width=True)
        df_show(out["table"].round(4))
    with tab2:
        gt = out["growth_table"].copy()
        df_show(gt.round(4))
        st.plotly_chart(bar_fig(gt, "Yếu tố", "Đóng góp vào tăng trưởng (%)", "Đóng góp tăng trưởng log bình quân"), use_container_width=True)
    with tab3:
        st.json({k: round(v, 3) for k, v in out["scenario2030"].items()})
        st.success("Kịch bản 2030 dùng D=30%, AI=100 nghìn DN số, H=35%, K và L tăng 6%/năm, TFP tăng 1,2%/năm.")

# -----------------------------------------------------------------------------
# Bai 2
# -----------------------------------------------------------------------------
elif page_id == 2:
    hero("Bài 2 — Phân bổ ngân sách số bằng quy hoạch tuyến tính", "DỄ", "Giải LP bằng scipy.optimize.linprog, phân tích shadow price và độ nhạy ngân sách.")
    with st.expander("Đề bài tóm tắt", expanded=True):
        st.markdown("""
        Tối đa hóa **Z = 0,85x₁ + 1,20x₂ + 0,95x₃ + 1,35x₄** với ngân sách tổng, sàn từng hạng mục và ràng buộc AI+R&D ≥ 35% tổng đầu tư.
        """)
    B = st.slider("Ngân sách tổng B (nghìn tỷ VND)", 80, 160, 100, 10)
    min_h = st.slider("Sàn nhân lực số x₃", 20, 50, 20, 5)
    res = core.solve_bai2_lp(B, min_h)
    if not res["success"]:
        st.error(res["message"])
    else:
        metric_cards([
            ("Z*", f"{res['Z']:.2f}", "GDP kỳ vọng tăng thêm"),
            ("Tổng chi", f"{res['x'].sum():.1f}", "nghìn tỷ VND"),
            ("AI + R&D", f"{res['x'].iloc[1] + res['x'].iloc[3]:.1f}", "x₂ + x₄"),
            ("Tỷ trọng chiến lược", f"{100*(res['x'].iloc[1]+res['x'].iloc[3])/res['x'].sum():.1f}%", "tối thiểu 35%"),
        ])
        df_show(res["x"].rename("Phân bổ tối ưu").reset_index().rename(columns={"index":"Hạng mục"}))
        st.plotly_chart(bar_fig(res["x"].reset_index().rename(columns={"index":"Hạng mục",0:"Giá trị"}), "Hạng mục", "Giá trị", "Phân bổ ngân sách tối ưu"), use_container_width=True)
        if res["dual"] is not None:
            st.subheader("Giá đối ngẫu / Shadow price")
            df_show(res["dual"].rename("Shadow price").reset_index().rename(columns={"index":"Ràng buộc"}).round(4))
    all2 = core.bai2_all()
    st.subheader("Độ nhạy ngân sách 100–120–140")
    df_show(all2["sensitivity"].round(3))
    st.plotly_chart(line_fig(all2["sensitivity"], "Ngân sách", ["Z*"], "Đường cong Z*(B)"), use_container_width=True)
    st.subheader("Kịch bản x₃ ≥ 30")
    h30 = all2["h30"]
    if h30["success"]:
        st.write(f"Bài toán vẫn khả thi. Z* = **{h30['Z']:.2f}**.")
        df_show(h30["x"].rename("Phân bổ").reset_index().rename(columns={"index":"Hạng mục"}))

# -----------------------------------------------------------------------------
# Bai 3
# -----------------------------------------------------------------------------
elif page_id == 3:
    hero("Bài 3 — Chỉ số ưu tiên ngành Priorityᵢ", "DỄ", "Chuẩn hóa min-max, tính Priority, phân tích độ nhạy trọng số AI Readiness.")
    with st.expander("Đề bài tóm tắt", expanded=True):
        st.markdown("""
        Priorityᵢ kết hợp tăng trưởng, năng suất, lan tỏa, xuất khẩu, việc làm, AI readiness và rủi ro tự động hóa. Risk được xử lý theo hướng **rủi ro thấp là tốt**.
        """)
    scheme = st.selectbox("Bộ trọng số", ["default", "growth", "inclusive"], format_func=lambda x: {"default":"Mặc định", "growth":"Định hướng tăng trưởng", "inclusive":"Định hướng bao trùm"}[x])
    w_ai = st.slider("Trọng số AI Readiness a₆", 0.05, 0.40, 0.20, 0.05)
    out = core.bai3_priority(ai_weight=w_ai, scheme=scheme)
    metric_cards([
        ("Top 1", out["ranking"].iloc[0]["sector_name_vi"], f"Priority={out['ranking'].iloc[0]['Priority']:.3f}"),
        ("Top 2", out["ranking"].iloc[1]["sector_name_vi"], f"Priority={out['ranking'].iloc[1]['Priority']:.3f}"),
        ("Top 3", out["ranking"].iloc[2]["sector_name_vi"], f"Priority={out['ranking'].iloc[2]['Priority']:.3f}"),
        ("Số ngành", "10", "dữ liệu sector 2024"),
    ])
    tab1, tab2, tab3 = st.tabs(["Xếp hạng", "Ma trận chuẩn hóa", "Độ nhạy"])
    with tab1:
        df_show(out["ranking"].round(4))
        st.plotly_chart(bar_fig(out["ranking"].sort_values("Priority"), "Priority", "sector_name_vi", "Priority theo ngành", orientation="h"), use_container_width=True)
        st.write("**Trọng số đang dùng**")
        df_show(out["weights"].rename("weight").reset_index().rename(columns={"index":"Tiêu chí"}).round(4))
    with tab2:
        df_show(out["normalized"].round(3))
    with tab3:
        sens = core.bai3_sensitivity()
        heat = sens.pivot(index="Ngành", columns="w_AI", values="Rank")
        fig = px.imshow(heat, text_auto=True, aspect="auto", template=PLOTLY_TEMPLATE, title="Heatmap thứ hạng khi thay đổi trọng số AI")
        st.plotly_chart(fig, use_container_width=True)
        df_show(sens.sort_values(["w_AI", "Rank"]))

# -----------------------------------------------------------------------------
# Bai 4
# -----------------------------------------------------------------------------
elif page_id == 4:
    hero("Bài 4 — LP phân bổ ngân sách số theo vùng-hạng mục", "TRUNG BÌNH", "Giải ma trận 6×4, so sánh có/không ràng buộc công bằng vùng miền.")
    with st.expander("Đề bài tóm tắt", expanded=True):
        st.markdown("""
        Phân bổ 50.000 tỷ VND cho 6 vùng và 4 hạng mục I, D, AI, H. Ràng buộc gồm ngân sách tổng, sàn/trần vùng, sàn nhân lực và công bằng số hóa.
        """)
    fair = st.toggle("Bật ràng buộc công bằng C5", value=True)
    res = core.solve_bai4_lp(fairness=fair)
    nofair = core.solve_bai4_lp(fairness=False)
    if res["success"]:
        alloc = res["allocation"]
        metric_cards([
            ("Z*", f"{res['Z']:,.1f}", "GDP gain kỳ vọng"),
            ("Tổng ngân sách", f"{alloc.values.sum():,.0f}", "tỷ VND"),
            ("Vùng nhận nhiều nhất", alloc.sum(axis=1).idxmax(), f"{alloc.sum(axis=1).max():,.0f}"),
            ("Chi phí công bằng", f"{(nofair['Z']-res['Z']):,.1f}", "so với bỏ C5"),
        ])
        df_show(alloc.round(2), hide_index=False)
        if res.get("slack") is not None and float(res["slack"].sum()) > 1e-6:
            st.warning("Với đúng tham số PDF (γ=0,002; λ=0,7; trần vùng 12.000), ràng buộc công bằng C5 bị căng/khó khả thi. App dùng soft fairness slack để vẫn có nghiệm phân bổ và hiển thị mức thiếu hụt bên dưới.")
            df_show(res["slack"].rename("Slack C5").reset_index().rename(columns={"index":"Vùng"}).round(3))
        fig = px.imshow(alloc, text_auto='.0f', aspect="auto", template=PLOTLY_TEMPLATE, title="Heatmap phân bổ tối ưu 6×4")
        st.plotly_chart(fig, use_container_width=True)
        comp = pd.DataFrame({"Có C5": alloc.sum(axis=1), "Không C5": nofair["allocation"].sum(axis=1)})
        st.plotly_chart(px.bar(comp, barmode="group", template=PLOTLY_TEMPLATE, title="So sánh tổng ngân sách vùng"), use_container_width=True)
        st.info(f"Nếu bỏ công bằng C5, Z* = {nofair['Z']:,.1f}. Chênh lệch = {nofair['Z']-res['Z']:,.1f} tỷ VND GDP gain.")
    else:
        st.error(res["message"])

# -----------------------------------------------------------------------------
# Bai 5
# -----------------------------------------------------------------------------
elif page_id == 5:
    hero("Bài 5 — MIP lựa chọn dự án chuyển đổi số", "TRUNG BÌNH", "Chọn tập dự án tối ưu với ràng buộc ngân sách, loại trừ, tiên quyết và số lượng dự án.")
    with st.expander("Đề bài tóm tắt", expanded=True):
        st.markdown("""
        Chọn trong 15 dự án, tối đa hóa NPV lợi ích, ngân sách 5 năm, ngân sách năm 1–2, ràng buộc y₁+y₂≤1, y₈≤y₁₂, y₁₃≤y₁₂, bắt buộc P14, số dự án 7–11.
        """)
    budget = st.slider("Ngân sách 5 năm", 70000, 110000, 80000, 5000)
    force = st.toggle("Bắt buộc có cả P1 và P2", value=False)
    expected = st.toggle("Tối đa hóa lợi ích kỳ vọng theo xác suất hoàn thành", value=False)
    res = core.solve_bai5(budget=budget, force_p1_p2=force, expected=expected)
    if not res["success"]:
        st.error(res["message"])
    else:
        metric_cards([
            ("Tổng lợi ích", f"{res['Z']:,.0f}", "tỷ VND"),
            ("Tổng chi phí", f"{res['cost']:,.0f}", "tỷ VND"),
            ("NPV/chi phí", f"{res['npv_per_cost']:.2f}", "hiệu suất"),
            ("Số dự án", f"{len(res['chosen_ids'])}", ", ".join('P'+str(i) for i in res['chosen_ids'])),
        ])
        df_show(res["chosen"][["id", "name", "sector", "cost", "benefit", "year12", "year35"]])
        fig = px.bar(res["chosen"], x="name", y=["cost", "benefit"], barmode="group", template=PLOTLY_TEMPLATE, title="Chi phí và lợi ích các dự án được chọn")
        fig.update_layout(xaxis_tickangle=-35, height=500)
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("So sánh ngân sách 80.000 và 100.000")
    comp = []
    for B in [80000, 100000]:
        r = core.solve_bai5(budget=B)
        comp.append({"Ngân sách": B, "Z*": r["Z"], "Chi phí": r["cost"], "Dự án chọn": ", ".join('P'+str(i) for i in r["chosen_ids"])})
    df_show(pd.DataFrame(comp))

# -----------------------------------------------------------------------------
# Bai 6
# -----------------------------------------------------------------------------
elif page_id == 6:
    hero("Bài 6 — TOPSIS xếp hạng 6 vùng ưu tiên đầu tư AI", "TRUNG BÌNH", "Tính TOPSIS bằng trọng số chuyên gia, Entropy weights và phân tích độ nhạy w_AI.")
    with st.expander("Đề bài tóm tắt", expanded=True):
        st.markdown("""
        Tiêu chí: GRDP/người, FDI, Digital Index, AI Readiness, lao động đào tạo, R&D, Internet là lợi ích; Gini là chi phí.
        """)
    w_ai = st.slider("w_AI trong TOPSIS", 0.10, 0.40, 0.20, 0.05)
    out = core.bai6_topsis(ai_weight=w_ai)
    metric_cards([
        ("Top chuyên gia", out["expert"].iloc[0]["region_name_vi"], f"Score={out['expert'].iloc[0]['TOPSIS_score']:.3f}"),
        ("Top Entropy", out["entropy"].iloc[0]["region_name_vi"], f"Score={out['entropy'].iloc[0]['TOPSIS_score']:.3f}"),
        ("Số vùng", "6", "KT-XH Việt Nam"),
        ("Gini", "Cost", "chỉ số càng thấp càng tốt"),
    ])
    tab1, tab2, tab3 = st.tabs(["TOPSIS", "Entropy", "Độ nhạy"])
    with tab1:
        df_show(out["expert"].round(4))
        st.plotly_chart(bar_fig(out["expert"].sort_values("TOPSIS_score"), "TOPSIS_score", "region_name_vi", "TOPSIS trọng số chuyên gia", orientation="h"), use_container_width=True)
    with tab2:
        df_show(out["entropy"].round(4))
        df_show(out["weights_entropy"].rename("Entropy weight").reset_index().rename(columns={"index":"Tiêu chí"}).round(4))
    with tab3:
        sens = core.bai6_sensitivity()
        heat = sens.pivot(index="Vùng", columns="w_AI", values="Rank")
        st.plotly_chart(px.imshow(heat, text_auto=True, aspect="auto", template=PLOTLY_TEMPLATE, title="Heatmap thứ hạng khi thay đổi w_AI"), use_container_width=True)
        df_show(sens.sort_values(["w_AI", "Rank"]))

# -----------------------------------------------------------------------------
# Bai 7
# -----------------------------------------------------------------------------
elif page_id == 7:
    hero("Bài 7 — Tối ưu đa mục tiêu Pareto", "KHÁ KHÓ", "Mô phỏng tập nghiệm không bị trội cho 4 mục tiêu: GDP, bao trùm, môi trường, an ninh dữ liệu.")
    with st.expander("Đề bài tóm tắt", expanded=True):
        st.markdown("""
        Bốn mục tiêu: **max GDP gain**, **min bất bình đẳng phân bổ**, **min phát thải**, **min rủi ro an ninh dữ liệu ròng**. Nghiệm thỏa hiệp được chọn bằng TOPSIS/weighted compromise.

        Lưu ý kỹ thuật: C5 của Bài 4 với γ=0,002 và trần vùng 12.000 rất khó/không khả thi, nên phần Pareto giữ C1-C4 và đưa công bằng vào mục tiêu bất bình đẳng để mô phỏng đánh đổi.
        """)
    n = st.slider("Số phương án mô phỏng", 200, 1500, 700, 100)
    seed = st.number_input("Seed", value=42, step=1)
    out = cached_bai7(n, int(seed))
    pareto = out["pareto"]
    best = out["best"]
    metric_cards([
        ("Số phương án", f"{len(out['all'])}", "khả thi"),
        ("Số Pareto", f"{len(pareto)}", "không bị trội"),
        ("GDP nghiệm thỏa hiệp", f"{best['GDP_gain']:,.0f}", "tỷ VND"),
        ("Score", f"{best['Compromise_score']:.3f}", "TOPSIS compromise"),
    ])
    fig3d = px.scatter_3d(pareto, x="GDP_gain", y="Inequality_MAD", z="Emission", color="SecurityRisk", template=PLOTLY_TEMPLATE, title="Pareto 3D: GDP - Bao trùm - Môi trường")
    fig3d.update_layout(height=620)
    st.plotly_chart(fig3d, use_container_width=True)
    df_show(pareto.sort_values("Compromise_score", ascending=False).head(20).round(3))
    alloc = pd.DataFrame(out["best_allocation"], index=[core.REGION_NAMES[r] for r in core.REGIONS], columns=["I", "D", "AI", "H"])
    st.subheader("Phân bổ của nghiệm thỏa hiệp")
    st.plotly_chart(px.imshow(alloc, text_auto='.0f', aspect="auto", template=PLOTLY_TEMPLATE), use_container_width=True)
    df_show(alloc.round(2), hide_index=False)

# -----------------------------------------------------------------------------
# Bai 8
# -----------------------------------------------------------------------------
elif page_id == 8:
    hero("Bài 8 — Tối ưu động phân bổ liên thời gian 2026–2035", "KHÁ KHÓ", "Mô phỏng quỹ đạo K, D, AI, H, Y, C và so sánh chiến lược đầu tư.")
    with st.expander("Đề bài tóm tắt", expanded=True):
        st.markdown("""
        Mục tiêu phúc lợi liên thời gian ∑ρᵗU(Cₜ), hàm sản xuất Cobb-Douglas, động học K, D, AI, H và cú sốc Y năm 2028.
        """)
    strategy = st.selectbox("Chiến lược", ["balanced", "even", "front_load", "traditional", "ai_led", "inclusive"], format_func=lambda x: {"balanced":"Tối ưu cân bằng", "even":"Đầu tư trải đều", "front_load":"Front-load", "traditional":"Truyền thống", "ai_led":"AI dẫn dắt", "inclusive":"Bao trùm"}[x])
    shock = st.toggle("Cú sốc 2028: Y giảm 8%", value=False)
    rho = st.slider("Hệ số chiết khấu ρ", 0.85, 0.99, 0.97, 0.01)
    df = core.simulate_dynamic(strategy=strategy, shock=shock, rho_discount=rho)
    metric_cards([
        ("Y 2035", f"{df.iloc[-1]['Y']:,.1f}", "sản lượng mô phỏng"),
        ("C 2035", f"{df.iloc[-1]['C']:,.1f}", "tiêu dùng"),
        ("Welfare", f"{df.iloc[-1]['welfare_cum']:.2f}", "tích lũy chiết khấu"),
        ("AI 2035", f"{df.iloc[-1]['AI']:.1f}", "nghìn DN số"),
    ])
    st.plotly_chart(line_fig(df, "year", ["K", "D", "AI", "H"], "Quỹ đạo trạng thái K, D, AI, H"), use_container_width=True)
    st.plotly_chart(line_fig(df, "year", ["Y", "C"], "Sản lượng Y và tiêu dùng C"), use_container_width=True)
    st.subheader("So sánh chiến lược")
    comp = core.bai8_compare()
    df_show(comp.round(3))
    st.plotly_chart(bar_fig(comp, "Chiến lược", "Welfare", "Welfare tổng theo chiến lược"), use_container_width=True)

# -----------------------------------------------------------------------------
# Bai 9
# -----------------------------------------------------------------------------
elif page_id == 9:
    hero("Bài 9 — Tác động AI tới thị trường lao động Việt Nam", "KHÁ KHÓ", "Tính NetJob, ngưỡng đào tạo lại và mô phỏng luồng dịch chuyển lao động.")
    with st.expander("Đề bài tóm tắt", expanded=True):
        st.markdown("""
        NetJobᵢ = NewJobᴬᴵᵢ + UpgradeJobᵢ − DisplacedJobᵢ. Ràng buộc tốc độ tự động hóa không vượt quá năng lực đào tạo lại: **DisplacedJobᵢ ≤ RetrainingCapacityᵢ**.
        """)
    strict = st.toggle("Thêm ràng buộc không ngành nào mất quá 5% lao động", value=False)
    out = core.solve_bai9(no_large_loss=strict)
    if not out["success"]:
        st.error(out["message"])
    else:
        tbl = out["table"]
        metric_cards([
            ("Tổng NetJob", f"{out['Z']:,.0f}", "việc làm ròng"),
            ("Ngân sách dùng", f"{(tbl['x_AI']+tbl['x_H']).sum():,.0f}", "tỷ VND"),
            ("Đào tạo nhiều nhất", tbl.sort_values("x_H", ascending=False).iloc[0]["Ngành"], f"x_H={tbl['x_H'].max():,.0f}"),
            ("AI nhiều nhất", tbl.sort_values("x_AI", ascending=False).iloc[0]["Ngành"], f"x_AI={tbl['x_AI'].max():,.0f}"),
        ])
        df_show(tbl.round(2))
        st.plotly_chart(px.bar(tbl, x="Ngành", y=["NewJob", "UpgradeJob", "DisplacedJob", "NetJob"], barmode="group", template=PLOTLY_TEMPLATE, title="Việc làm tạo mới, nâng cấp, dịch chuyển và ròng"), use_container_width=True)
        st.subheader("Ngưỡng x_H ngành 2")
        xai2 = st.slider("Giả định x_AI ngành 2", 0, 30000, int(max(tbl.loc[1, "x_AI"], 1000)), 500)
        threshold = core.bai9_threshold_industry2(x_ai=float(xai2))
        st.json({k: round(v, 2) for k,v in threshold.items()})
        st.subheader("Sankey nhóm dễ bị tổn thương: ngành 1, 3, 4")
        vuln_idx = [0,2,3]
        labels = []
        for i in vuln_idx:
            labels.append(tbl.loc[i,"Ngành"] + " - Displaced")
        labels += ["Đào tạo lại", "Việc làm mới/nâng cấp", "Thất nghiệp ròng"]
        sources, targets, values = [], [], []
        retrain_node = len(vuln_idx); new_node = retrain_node+1; unemp_node = retrain_node+2
        for k,i in enumerate(vuln_idx):
            displaced = float(tbl.loc[i,"DisplacedJob"])
            retrained = min(displaced, float(tbl.loc[i,"RetrainingCapacity"]))
            unemp = max(displaced-retrained, 0)
            sources += [k, retrain_node]
            targets += [retrain_node, new_node]
            values += [max(retrained,0), max(retrained + tbl.loc[i,"NewJob"] + tbl.loc[i,"UpgradeJob"],0)]
            if unemp > 1e-6:
                sources.append(k); targets.append(unemp_node); values.append(unemp)
        fig = go.Figure(data=[go.Sankey(node=dict(label=labels), link=dict(source=sources, target=targets, value=values))])
        fig.update_layout(template=PLOTLY_TEMPLATE, height=500)
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# Bai 10 - updated standalone implementation merged into full app
# -----------------------------------------------------------------------------
elif page_id == 10:
    hero("Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn dưới bất định", "KHÓ", "First-stage / second-stage với 4 kịch bản, tính SP, EV, WS, VSS, EVPI và robust minimax regret.")
    st.success("ĐANG CHẠY ĐÚNG BÀI 10 — không còn nhảy về Bài 1")
    with st.expander("Đề bài tóm tắt", expanded=True):
        st.markdown("""
        - First-stage: phân bổ ban đầu **x = (I, D, AI, H)**, tổng tối đa **65 nghìn tỷ VND**.  
        - Second-stage: điều chỉnh theo kịch bản **yˢ**, dự phòng **15 nghìn tỷ VND**.  
        - Ràng buộc liên kết: **y_AIˢ ≤ 0,5 × x_H**.  
        - Mục tiêu: tối đa hóa GDP gain kỳ vọng, so sánh **SP / EV / WS**, tính **VSS** và **EVPI**.
        """)

    c1, c2, c3 = st.columns(3)
    with c1:
        first_budget = st.number_input("First-stage budget", 10.0, 100.0, 65.0, 1.0)
    with c2:
        reserve_budget = st.number_input("Second-stage reserve", 1.0, 50.0, 15.0, 1.0)
    with c3:
        step = st.selectbox("Độ mịn tối ưu H", [0.10, 0.05, 0.01], index=2)

    result = bai10_new.compute_ws_evpi_vss(first_budget, reserve_budget, step)
    robust = bai10_new.optimize_robust_minimax_regret(first_budget, reserve_budget, step)
    sp = result["sp"]
    ev = result["ev"]
    rb = robust["solution"]

    metric_cards([
        ("SP objective", f"{result['SP']:.2f}", "Stochastic solution"),
        ("EEV", f"{result['EEV']:.2f}", "Expected result of EV"),
        ("VSS", f"{result['VSS']:.4f}", "Value of stochastic solution"),
        ("EVPI", f"{result['EVPI']:.4f}", "Expected value of perfect information"),
    ])

    tab1, tab2, tab3, tab4 = st.tabs(["Kịch bản & hệ số", "SP / EV / Robust", "VSS - EVPI", "Mã Pyomo tham khảo"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Cây kịch bản")
            st.dataframe(fmt_df(bai10_new.scenario_dataframe(), 2), use_container_width=True)
        with col2:
            st.subheader("Hệ số β theo kịch bản")
            st.dataframe(fmt_df(bai10_new.beta_dataframe(), 2), use_container_width=True)

    with tab2:
        st.subheader("Quyết định first-stage")
        st.dataframe(fmt_df(bai10_new.first_stage_table(sp, ev, rb), 3), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Second-stage của SP")
            st.dataframe(fmt_df(bai10_new.recourse_table(sp), 3), use_container_width=True)
        with col2:
            st.subheader("Regret của Robust")
            regret_df = pd.DataFrame([{"Kịch bản": k, "Regret": v} for k, v in robust["regrets"].items()])
            st.dataframe(fmt_df(regret_df, 3), use_container_width=True)
        alloc = bai10_new.allocation_long_table(sp)
        fig = px.bar(
            alloc,
            x="Hạng mục",
            y="Ngân sách",
            color="Kịch bản",
            facet_col="Giai đoạn",
            barmode="group",
            template=PLOTLY_TEMPLATE,
            title="Phân bổ first-stage và second-stage của SP",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        det_rows = []
        for s, sol in result["deterministic"].items():
            row = {"Kịch bản": s, "Tên": bai10_new.SCENARIO_NAMES[s], "WS objective": sol.expected_value, "x_H": sol.h_value}
            row.update({f"x_{j}": sol.first_stage[j] for j in bai10_new.ITEMS})
            det_rows.append(row)
        det_df = pd.DataFrame(det_rows)
        st.dataframe(fmt_df(det_df, 3), use_container_width=True)
        metric_df = pd.DataFrame([
            {"Chỉ tiêu": "WS - Wait and See", "Giá trị": result["WS"]},
            {"Chỉ tiêu": "SP - Stochastic Solution", "Giá trị": result["SP"]},
            {"Chỉ tiêu": "EEV - Expected result of EV", "Giá trị": result["EEV"]},
            {"Chỉ tiêu": "VSS = SP - EEV", "Giá trị": result["VSS"]},
            {"Chỉ tiêu": "EVPI = WS - SP", "Giá trị": result["EVPI"]},
        ])
        st.plotly_chart(px.bar(metric_df, x="Chỉ tiêu", y="Giá trị", template=PLOTLY_TEMPLATE, title="So sánh WS, SP, EEV, VSS, EVPI"), use_container_width=True)
        st.markdown("""
        <div class='card'>Trong bộ tham số gốc, VSS có thể rất nhỏ hoặc bằng 0 nếu EV và SP cùng chọn một first-stage giống nhau. Đây không phải lỗi code mà là đặc điểm của hệ số/ràng buộc hiện tại. EVPI phản ánh giá trị thông tin hoàn hảo khi biết trước kịch bản.</div>
        """, unsafe_allow_html=True)

    with tab4:
        st.code("""
import pyomo.environ as pyo
m = pyo.ConcreteModel()
m.J = pyo.Set(initialize=['I','D','AI','H'])
m.S = pyo.Set(initialize=['s1','s2','s3','s4'])
m.p = pyo.Param(m.S, initialize={'s1':0.30,'s2':0.45,'s3':0.20,'s4':0.05})
m.beta = pyo.Param(m.J, initialize={'I':1.00,'D':1.10,'AI':1.25,'H':0.95})
beta_s = {('s1','I'):1.25,('s1','D'):1.35,('s1','AI'):1.55,('s1','H'):1.05,
          ('s2','I'):1.00,('s2','D'):1.10,('s2','AI'):1.25,('s2','H'):0.95,
          ('s3','I'):0.75,('s3','D'):0.85,('s3','AI'):0.90,('s3','H'):1.00,
          ('s4','I'):0.40,('s4','D'):0.50,('s4','AI'):0.55,('s4','H'):1.10}
m.beta_s = pyo.Param(m.S, m.J, initialize=beta_s)
m.x = pyo.Var(m.J, within=pyo.NonNegativeReals)
m.y = pyo.Var(m.S, m.J, within=pyo.NonNegativeReals)
m.budget1 = pyo.Constraint(expr=sum(m.x[j] for j in m.J) <= 65)
m.budget2 = pyo.Constraint(m.S, rule=lambda m,s: sum(m.y[s,j] for j in m.J) <= 15)
m.ai_link = pyo.Constraint(m.S, rule=lambda m,s: m.y[s,'AI'] <= 0.5*m.x['H'])
m.obj = pyo.Objective(
    expr=sum(m.beta[j]*m.x[j] for j in m.J) +
         sum(m.p[s]*sum(m.beta_s[s,j]*m.y[s,j] for j in m.J) for s in m.S),
    sense=pyo.maximize
)
# pyo.SolverFactory('glpk').solve(m)
        """, language="python")

# -----------------------------------------------------------------------------
# Bai 11 - updated standalone implementation merged into full app
# -----------------------------------------------------------------------------
elif page_id == 11:
    hero("Bài 11 — Q-learning cho chính sách kinh tế thích nghi", "KHÓ", "MDP 81 trạng thái, 5 hành động chính sách, reward cân bằng tăng trưởng - thất nghiệp - thất nghiệp - an ninh mạng - phát thải.".replace("thất nghiệp - thất nghiệp", "thất nghiệp"))
    st.success("ĐANG CHẠY ĐÚNG BÀI 11 — không còn nhảy về Bài 1")
    with st.expander("Đề bài tóm tắt", expanded=True):
        st.markdown("""
        Trạng thái gồm 4 chiều rời rạc: **GDP growth, Digital index, AI capacity, Unemployment risk**.  
        Hành động gồm 5 gói phân bổ ngân sách: truyền thống, cân bằng, số hóa nhanh, AI dẫn dắt, bao trùm.  
        Phần thưởng: **R = w₁ΔGDP − w₂Unemployment − w₃CyberRisk − w₄Emission**, với **w = (40, 25, 20, 15)**.
        """)

    st.subheader("Bảng hành động chính sách")
    st.dataframe(fmt_df(bai11_new.action_table(), 2), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        episodes = st.slider("Số episode", 500, 10000, 3000, 500)
    with c2:
        alpha_rl = st.slider("α learning rate", 0.01, 0.50, 0.10, 0.01)
    with c3:
        gamma_rl = st.slider("γ discount", 0.50, 0.99, 0.95, 0.01)
    with c4:
        seed_rl = st.number_input("Seed", 0, 9999, 42, 1)

    @st.cache_data(show_spinner="Đang train Q-learning...")
    def cached_train_bai11(episodes, alpha, gamma, seed):
        return bai11_new.train_q_learning(episodes=int(episodes), alpha=float(alpha), gamma=float(gamma), seed=int(seed))

    trained = cached_train_bai11(episodes, alpha_rl, gamma_rl, seed_rl)
    Q = trained["Q"]
    rewards = pd.DataFrame({"Episode": np.arange(len(trained["rewards"])), "Reward": trained["rewards"], "Smoothed": trained["smooth_rewards"]})

    metric_cards([
        ("Mean reward 100 ep cuối", f"{np.mean(trained['rewards'][-100:]):.2f}", "learning curve"),
        ("Best episode reward", f"{np.max(trained['rewards']):.2f}", "episode tốt nhất"),
        ("Trạng thái", "81", "3⁴"),
        ("Hành động", "5", "a0–a4"),
    ])

    tab1, tab2, tab3 = st.tabs(["Learning curve", "So sánh chính sách", "Policy map"])
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rewards["Episode"], y=rewards["Reward"], mode="lines", name="Per episode", opacity=0.35))
        fig.add_trace(go.Scatter(x=rewards["Episode"], y=rewards["Smoothed"], mode="lines", name="Smoothed"))
        fig.update_layout(template=PLOTLY_TEMPLATE, title="Learning curve Q-learning", xaxis_title="Episode", yaxis_title="Cumulative reward")
        st.plotly_chart(fig, use_container_width=True)
        action_counts = pd.DataFrame({"Hành động": [bai11_new.ACTION_NAMES[i] for i in range(5)], "Số lần được chọn": trained["actions_count"]})
        st.plotly_chart(px.bar(action_counts, x="Hành động", y="Số lần được chọn", template=PLOTLY_TEMPLATE, title="Tần suất hành động trong quá trình huấn luyện"), use_container_width=True)

    with tab2:
        comp = bai11_new.compare_policies(Q, episodes=250, seed=int(seed_rl) + 1)
        st.dataframe(fmt_df(comp, 3), use_container_width=True)
        st.plotly_chart(px.bar(comp, x="Chính sách", y="Mean cumulative reward", template=PLOTLY_TEMPLATE, title="So sánh π* với chính sách rule-based"), use_container_width=True)
        path = bai11_new.simulate_one_episode_policy(Q=Q, seed=int(seed_rl) + 5)
        st.plotly_chart(px.line(path, x="t", y=["D", "AI", "H"], markers=True, template=PLOTLY_TEMPLATE, title="Một episode mô phỏng theo chính sách học được"), use_container_width=True)

    with tab3:
        st.caption("Chọn lát cắt trạng thái để xem Q-learning khuyến nghị hành động nào.")
        col1, col2 = st.columns(2)
        with col1:
            g_state = st.selectbox("GDP growth state", [0, 1, 2], index=1, format_func=lambda x: ["low", "medium", "high"][x])
        with col2:
            u_state = st.selectbox("Unemployment risk state", [0, 1, 2], index=1, format_func=lambda x: ["low", "medium", "high"][x])
        sl = bai11_new.policy_slice(Q, g_state, u_state)
        pivot = sl.pivot(index="Digital state", columns="AI state", values="Action")
        st.plotly_chart(px.imshow(pivot, text_auto=True, aspect="auto", template=PLOTLY_TEMPLATE, title="Policy heatmap: số hành động được chọn"), use_container_width=True)
        st.dataframe(bai11_new.policy_table_for_states(Q), use_container_width=True)

# -----------------------------------------------------------------------------
# Bai 12 - updated standalone implementation merged into full app
# -----------------------------------------------------------------------------
elif page_id == 12:
    hero("Bài 12 — AIDEOM-VN Dashboard tích hợp", "KHÓ", "Nguyên mẫu tích hợp 6 module M1-M6, so sánh 5 kịch bản chính sách và đưa ra khuyến nghị tổng hợp.")
    st.success("ĐANG CHẠY ĐÚNG BÀI 12 — không còn nhảy về Bài 1")
    with st.expander("Đề bài tóm tắt", expanded=True):
        st.markdown("""
        Tích hợp **M1 dự báo kinh tế, M2 sẵn sàng số, M3 tối ưu phân bổ, M4 lao động, M5 rủi ro, M6 dashboard ra quyết định**.  
        So sánh 5 kịch bản: **S1 Truyền thống, S2 Số hóa nhanh, S3 AI dẫn dắt, S4 Bao trùm số, S5 Tối ưu cân bằng**.
        """)

    outputs = bai12_new.run_integrated_model()
    scen = outputs["scenario_table"]
    summary = outputs["summary"]
    paths = outputs["paths"]
    risks = outputs["risks"]

    tab1, tab2, tab3, tab4 = st.tabs(["Tổng quan M1-M6", "So sánh 5 kịch bản", "Rủi ro & việc làm", "Khuyến nghị"])
    with tab1:
        st.subheader("Thiết kế hệ thống AIDEOM-VN")
        st.dataframe(outputs["modules"], use_container_width=True)
        st.subheader("Năm kịch bản chính sách")
        st.dataframe(fmt_df(scen, 2), use_container_width=True)

    with tab2:
        top = summary.iloc[0]
        metric_cards([
            ("GDP 2030 cao nhất", f"{top['GDP 2030']:.1f}", str(top["Kịch bản"])),
            ("Rủi ro thấp nhất", str(summary.sort_values("Risk_index").iloc[0]["Kịch bản"]), "Risk_index min"),
            ("Việc làm tốt nhất", str(summary.sort_values("Net_jobs_index", ascending=False).iloc[0]["Kịch bản"]), "Net_jobs_index max"),
            ("Kịch bản", "5", "S1–S5"),
        ])
        st.dataframe(fmt_df(summary, 2), use_container_width=True)
        st.plotly_chart(px.line(paths, x="Năm", y="GDP", color="Kịch bản", markers=True, template=PLOTLY_TEMPLATE, title="Dự báo GDP 2025-2030 theo 5 kịch bản"), use_container_width=True)
        st.plotly_chart(px.bar(summary, x="Kịch bản", y="Tăng GDP 2025-2030 (%)", color="Tên", template=PLOTLY_TEMPLATE, title="Tăng trưởng GDP tích lũy 2025-2030"), use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.bar(summary, x="Kịch bản", y="Net_jobs_index", color="Tên", template=PLOTLY_TEMPLATE, title="Chỉ số việc làm ròng"), use_container_width=True)
        with col2:
            st.plotly_chart(px.bar(summary, x="Kịch bản", y="Risk_index", color="Tên", template=PLOTLY_TEMPLATE, title="Chỉ số rủi ro tổng hợp"), use_container_width=True)
        st.plotly_chart(px.bar(risks, x="Kịch bản", y="Điểm", color="Loại rủi ro", barmode="group", template=PLOTLY_TEMPLATE, title="Cấu phần rủi ro: Cyber - Environment - Dependency"), use_container_width=True)
        radar_scen = st.selectbox("Chọn kịch bản radar", summary["Kịch bản"].tolist(), index=0)
        row = summary.loc[summary["Kịch bản"] == radar_scen].iloc[0]
        cats = ["GDP_2030_index", "Readiness", "Net_jobs_index", "Risk_index"]
        fig = go.Figure(data=go.Scatterpolar(r=[row[c] for c in cats], theta=cats, fill="toself"))
        fig.update_layout(template=PLOTLY_TEMPLATE, title=f"Radar kịch bản {radar_scen}", polar=dict(radialaxis=dict(visible=True)))
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown(f"<div class='card'>{bai12_new.recommendation(summary)}</div>", unsafe_allow_html=True)
        st.markdown("""
        **Gợi ý viết thảo luận chính sách:**  
        - Không nên chọn duy nhất kịch bản có GDP cao nhất nếu rủi ro an ninh mạng, phụ thuộc công nghệ hoặc sốc lao động quá lớn.  
        - S5 là phương án thỏa hiệp phù hợp để trình bày trong báo cáo vì kết hợp logic của stochastic planning, Q-learning và dashboard tích hợp.  
        - Cần bổ sung ràng buộc an sinh xã hội, minh bạch dữ liệu và trách nhiệm giải trình khi dùng AI hỗ trợ chính sách.
        """)
