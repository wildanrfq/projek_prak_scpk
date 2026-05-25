import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# page config

st.set_page_config(page_title="SPK Fuzzy Irigasi", layout="wide")


# load dataset
# gunakan cache_data agar dataset tidak selalu di read setiap rerun
@st.cache_data
def load_data():
    df = pd.read_csv("irrigation_prediction.csv")
    return df


df = load_data()

# sidebar navigation

menu = st.sidebar.radio(
    "Navigasi", ["Dashboard", "Dataset", "SPK Fuzzy", "Visualisasi", "Profil Kelompok"]
)


# sistem fuzzy
# pakai cache_resource agar system fuzzy tidak dibangun ulang setiap rerun
@st.cache_resource
def build_fuzzy_system():
    suhu = ctrl.Antecedent(np.arange(0, 51, 1), "suhu")
    kel_tanah = ctrl.Antecedent(np.arange(0, 101, 1), "kel_tanah")
    curah_hujan = ctrl.Antecedent(np.arange(0, 101, 1), "curah_hujan")
    kel_udara = ctrl.Antecedent(np.arange(0, 101, 1), "kel_udara")
    kebutuhan = ctrl.Consequent(np.arange(0, 101, 1), "kebutuhan")

    suhu["rendah"] = fuzz.trimf(suhu.universe, [0, 0, 20])
    suhu["sedang"] = fuzz.trimf(suhu.universe, [15, 25, 35])
    suhu["tinggi"] = fuzz.trimf(suhu.universe, [30, 50, 50])

    kel_tanah["kering"] = fuzz.trimf(kel_tanah.universe, [0, 0, 40])
    kel_tanah["lembab"] = fuzz.trimf(kel_tanah.universe, [30, 50, 70])
    kel_tanah["basah"] = fuzz.trimf(kel_tanah.universe, [60, 100, 100])

    curah_hujan["rendah"] = fuzz.trimf(curah_hujan.universe, [0, 0, 30])
    curah_hujan["sedang"] = fuzz.trimf(curah_hujan.universe, [20, 50, 80])
    curah_hujan["tinggi"] = fuzz.trimf(curah_hujan.universe, [70, 100, 100])

    kel_udara["rendah"] = fuzz.trimf(kel_udara.universe, [0, 0, 40])
    kel_udara["sedang"] = fuzz.trimf(kel_udara.universe, [30, 50, 70])
    kel_udara["tinggi"] = fuzz.trimf(kel_udara.universe, [60, 100, 100])

    kebutuhan["sedikit"] = fuzz.trimf(kebutuhan.universe, [0, 0, 40])
    kebutuhan["sedang"] = fuzz.trimf(kebutuhan.universe, [30, 50, 70])
    kebutuhan["banyak"] = fuzz.trimf(kebutuhan.universe, [60, 100, 100])

    rules = [
        ctrl.Rule(
            suhu["tinggi"] & kel_tanah["kering"] & curah_hujan["rendah"],
            kebutuhan["banyak"],
        ),
        ctrl.Rule(suhu["tinggi"] & kel_tanah["kering"], kebutuhan["banyak"]),
        ctrl.Rule(suhu["sedang"] & kel_tanah["lembab"], kebutuhan["sedang"]),
        ctrl.Rule(suhu["rendah"] & kel_tanah["lembab"], kebutuhan["sedikit"]),
        ctrl.Rule(curah_hujan["tinggi"], kebutuhan["sedikit"]),
        ctrl.Rule(kel_tanah["basah"], kebutuhan["sedikit"]),
        ctrl.Rule(suhu["tinggi"] & kel_udara["rendah"], kebutuhan["banyak"]),
        ctrl.Rule(
            suhu["sedang"] & kel_udara["sedang"] & curah_hujan["sedang"],
            kebutuhan["sedang"],
        ),
        ctrl.Rule(suhu["rendah"] & curah_hujan["tinggi"], kebutuhan["sedikit"]),
        ctrl.Rule(kel_tanah["kering"] & curah_hujan["rendah"], kebutuhan["banyak"]),
    ]

    irrigation_ctrl = ctrl.ControlSystem(rules)
    simulation = ctrl.ControlSystemSimulation(irrigation_ctrl)

    return {
        "suhu": suhu,
        "kel_tanah": kel_tanah,
        "curah_hujan": curah_hujan,
        "kel_udara": kel_udara,
        "kebutuhan": kebutuhan,
        "simulation": simulation,
    }


def plot_membership(var, title, input_val=None):
    """Plot kurva keanggotaan menggunakan plt.plot() sesuai materi pertemuan 3."""
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["#2196F3", "#4CAF50", "#FF5722"]

    for i, (label, mf_obj) in enumerate(var.terms.items()):
        ax.plot(
            var.universe,
            mf_obj.mf,
            color=colors[i % len(colors)],
            linewidth=2,
            label=label,
            linestyle="-",
        )

    if input_val is not None:
        ax.plot(
            [input_val, input_val],
            [0, 1],
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Input: {input_val:.1f}",
        )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Nilai")
    ax.set_ylabel("Derajat Keanggotaan")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="-.", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_output_membership(kebutuhan_var, hasil):
    """Plot output membership functions + garis defuzzifikasi."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = {"sedikit": "#2196F3", "sedang": "#4CAF50", "banyak": "#FF5722"}

    for label, mf_obj in kebutuhan_var.terms.items():
        ax.plot(
            kebutuhan_var.universe,
            mf_obj.mf,
            color=colors.get(label, "gray"),
            linewidth=2,
            label=label,
            linestyle="--",
        )

    # garis defuzzifikasi

    ax.plot(
        [hasil, hasil],
        [0, 1.05],
        color="red",
        linestyle="-",
        linewidth=2,
        label=f"Defuzzifikasi: {hasil:.2f}",
    )

    ax.set_title("Output Fuzzy - Kebutuhan Air", fontweight="bold")
    ax.set_xlabel("Kebutuhan Air (%)")
    ax.set_ylabel("Derajat Keanggotaan")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="-.", alpha=0.3)
    fig.tight_layout()
    return fig


def get_kategori(nilai):
    if nilai < 40:
        return "Sedikit", "Kebutuhan air rendah. Irigasi minimal atau tidak perlu."
    elif nilai < 70:
        return "Sedang", "Kebutuhan air sedang. Lakukan irigasi secukupnya."
    else:
        return "Banyak", "Kebutuhan air tinggi. Segera lakukan irigasi intensif."


# dashboard

if menu == "Dashboard":

    st.title("Sistem Pendukung Keputusan Irigasi Berbasis Fuzzy")

    st.markdown("""
    Sistem ini digunakan untuk menentukan **kebutuhan air irigasi** berdasarkan kondisi
    lingkungan dan pertanian menggunakan metode **Fuzzy Mamdani**.
    """)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Data", f"{df.shape[0]:,} baris")
    with col2:
        st.metric("Jenis Tanaman", df["Crop_Type"].nunique())
    with col3:
        st.metric("Jenis Tanah", df["Soil_Type"].nunique())
    with col4:
        low_pct = round((df["Irrigation_Need"] == "Low").mean() * 100, 1)
        st.metric("Data Kebutuhan Rendah", f"{low_pct}%")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Variabel Input")
        st.markdown("""
        | No | Variabel | Satuan | Range |
        |----|----------|--------|-------|
        | 1 | Suhu Udara | C | 0 - 50 |
        | 2 | Kelembaban Tanah | % | 0 - 100 |
        | 3 | Curah Hujan (norm.) | % | 0 - 100 |
        | 4 | Kelembaban Udara | % | 0 - 100 |
        """)

    with col_b:
        st.subheader("Variabel Output")
        st.markdown("""
        | Kategori | Nilai Fuzzy | Keterangan |
        |----------|------------|------------|
        | Sedikit | 0 - 40 | Tidak perlu irigasi |
        | Sedang | 30 - 70 | Irigasi secukupnya |
        | Banyak | 60 - 100 | Irigasi intensif |
        """)

    st.divider()
    st.subheader("Alur Sistem Fuzzy Mamdani")

    st.markdown("""
    ```
    Input --> Fuzzifikasi --> Evaluasi Rules --> Agregasi --> Defuzzifikasi --> Output
    ```
    1. **Fuzzifikasi** : Nilai input dikonversi ke derajat keanggotaan
    2. **Evaluasi Rules** : 10 aturan fuzzy dievaluasi (metode MIN)
    3. **Agregasi** : Hasil semua rule digabung (metode MAX)
    4. **Defuzzifikasi** : Area teraktivasi diubah ke nilai crisp (metode Centroid)
    """)


# page dataset

elif menu == "Dataset":

    st.title("Dataset Irigasi")

    st.write(
        f"Dataset terdiri dari **{df.shape[0]:,} baris** dan **{df.shape[1]} kolom**."
    )

    st.dataframe(df.head(100), width="stretch")
    st.caption("Menampilkan 100 baris pertama dari dataset.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Statistik Deskriptif")
        st.dataframe(df.describe().round(2), width="stretch")

    with col2:
        st.subheader("Distribusi Kelas Output")
        dist = df["Irrigation_Need"].value_counts().reset_index()
        dist.columns = ["Kategori", "Jumlah"]
        dist["Persentase (%)"] = (dist["Jumlah"] / len(df) * 100).round(2)
        st.dataframe(dist, width="stretch")

    st.divider()
    st.subheader("Informasi Kolom")
    info_df = pd.DataFrame(
        {
            "Kolom": df.columns,
            "Tipe Data": df.dtypes.astype(str).values,
            "Nilai Unik": [df[c].nunique() for c in df.columns],
            "Missing Values": df.isnull().sum().values,
        }
    )
    st.dataframe(info_df, width="stretch")


# spk fuzzy page

elif menu == "SPK Fuzzy":

    st.title("Perhitungan SPK Fuzzy Mamdani")

    st.info(
        "Curah hujan dikonversi ke skala 0-100 (normalisasi dari nilai aktual dataset 0-2500 mm).",
    )

    st.subheader("Input Parameter")

    col1, col2 = st.columns(2)

    with col1:
        temperature = st.slider("Suhu Udara (C)", 0.0, 50.0, 30.0, step=0.5)
        soil_moisture = st.slider("Kelembaban Tanah (%)", 0.0, 100.0, 50.0, step=0.5)
        rainfall_raw = st.slider("Curah Hujan (mm/hari)", 0.0, 2500.0, 500.0, step=10.0)
        rainfall_norm = round((rainfall_raw / 2500.0) * 100, 2)
        st.caption(f"Curah hujan ternormalisasi: {rainfall_norm:.1f} / 100")

    with col2:
        humidity = st.slider("Kelembaban Udara (%)", 0.0, 100.0, 60.0, step=0.5)
        crop_type = st.selectbox("Jenis Tanaman", sorted(df["Crop_Type"].unique()))
        soil_type = st.selectbox("Jenis Tanah", sorted(df["Soil_Type"].unique()))
        growth_stage = st.selectbox(
            "Tahap Pertumbuhan", sorted(df["Crop_Growth_Stage"].unique())
        )

    st.divider()

    fuzzy = build_fuzzy_system()

    # kurva keanggotaan

    st.subheader("Kurva Keanggotaan (Membership Functions)")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Suhu", "Kel. Tanah", "Curah Hujan", "Kel. Udara", "Output"]
    )

    with tab1:
        st.pyplot(plot_membership(fuzzy["suhu"], "Suhu Udara", temperature))
    with tab2:
        st.pyplot(
            plot_membership(fuzzy["kel_tanah"], "Kelembaban Tanah", soil_moisture)
        )
    with tab3:
        st.pyplot(
            plot_membership(fuzzy["curah_hujan"], "Curah Hujan (norm.)", rainfall_norm)
        )
    with tab4:
        st.pyplot(plot_membership(fuzzy["kel_udara"], "Kelembaban Udara", humidity))
    with tab5:
        st.pyplot(plot_membership(fuzzy["kebutuhan"], "Kebutuhan Air (Output)"))

    st.divider()

    # tabel rules

    st.subheader("Basis Aturan Fuzzy (Rule Base)")

    rules_df = pd.DataFrame(
        [
            [
                "R1",
                "Suhu TINGGI",
                "Kel.Tanah KERING",
                "Hujan RENDAH",
                "-",
                "Kebutuhan BANYAK",
            ],
            ["R2", "Suhu TINGGI", "Kel.Tanah KERING", "-", "-", "Kebutuhan BANYAK"],
            ["R3", "Suhu SEDANG", "Kel.Tanah LEMBAB", "-", "-", "Kebutuhan SEDANG"],
            ["R4", "Suhu RENDAH", "Kel.Tanah LEMBAB", "-", "-", "Kebutuhan SEDIKIT"],
            ["R5", "-", "-", "Hujan TINGGI", "-", "Kebutuhan SEDIKIT"],
            ["R6", "-", "Kel.Tanah BASAH", "-", "-", "Kebutuhan SEDIKIT"],
            ["R7", "Suhu TINGGI", "-", "-", "Kel.Udara RENDAH", "Kebutuhan BANYAK"],
            [
                "R8",
                "Suhu SEDANG",
                "-",
                "Hujan SEDANG",
                "Kel.Udara SEDANG",
                "Kebutuhan SEDANG",
            ],
            ["R9", "Suhu RENDAH", "-", "Hujan TINGGI", "-", "Kebutuhan SEDIKIT"],
            ["R10", "-", "Kel.Tanah KERING", "Hujan RENDAH", "-", "Kebutuhan BANYAK"],
        ],
        columns=["Rule", "Suhu", "Kel. Tanah", "Curah Hujan", "Kel. Udara", "Output"],
    )

    st.dataframe(rules_df, width="stretch", hide_index=True)

    st.divider()

    # eksekusi

    if st.button("Hitung Kebutuhan Air", type="primary"):

        sim = fuzzy["simulation"]

        try:
            sim.input["suhu"] = temperature
            sim.input["kel_tanah"] = soil_moisture
            sim.input["curah_hujan"] = rainfall_norm
            sim.input["kel_udara"] = humidity

            sim.compute()

            hasil = sim.output["kebutuhan"]
            kategori, keterangan = get_kategori(hasil)

            st.success(f"Nilai Defuzzifikasi (Centroid): **{hasil:.4f}**")

            col_r1, col_r2 = st.columns(2)

            with col_r1:
                st.subheader("Hasil Keputusan")
                if kategori == "Sedikit":
                    st.success(f"Kategori: **{kategori}**\n\n{keterangan}")
                elif kategori == "Sedang":
                    st.warning(f"Kategori: **{kategori}**\n\n{keterangan}")
                else:
                    st.error(f"Kategori: **{kategori}**\n\n{keterangan}")

            with col_r2:
                st.subheader("Tabel Defuzzifikasi")

                universe = fuzzy["kebutuhan"].universe
                mf_sedikit = fuzz.trimf(universe, [0, 0, 40])
                mf_sedang = fuzz.trimf(universe, [30, 50, 70])
                mf_banyak = fuzz.trimf(universe, [60, 100, 100])

                deg_sedikit = round(
                    float(fuzz.interp_membership(universe, mf_sedikit, hasil)), 4
                )
                deg_sedang = round(
                    float(fuzz.interp_membership(universe, mf_sedang, hasil)), 4
                )
                deg_banyak = round(
                    float(fuzz.interp_membership(universe, mf_banyak, hasil)), 4
                )

                defuzz_df = pd.DataFrame(
                    {
                        "Variabel Linguistik": ["Sedikit", "Sedang", "Banyak"],
                        "Range": ["[0, 0, 40]", "[30, 50, 70]", "[60, 100, 100]"],
                        "Derajat Keanggotaan": [deg_sedikit, deg_sedang, deg_banyak],
                        "Nilai Crisp (x)": [hasil, hasil, hasil],
                    }
                )
                st.dataframe(defuzz_df, width="stretch", hide_index=True)

            st.subheader("Visualisasi Output Fuzzy")
            fig_out = plot_output_membership(fuzzy["kebutuhan"], hasil)
            st.pyplot(fig_out)

            st.subheader("Tabel Hasil Perangkingan")

            ranking_df = pd.DataFrame(
                {
                    "Alternatif": [
                        "Irigasi Banyak",
                        "Irigasi Sedang",
                        "Irigasi Sedikit",
                    ],
                    "Nilai Fuzzy": [
                        hasil if kategori == "Banyak" else 70,
                        hasil if kategori == "Sedang" else 50,
                        hasil if kategori == "Sedikit" else 20,
                    ],
                    "Status": [
                        "DIREKOMENDASIKAN" if kategori == "Banyak" else "-",
                        "DIREKOMENDASIKAN" if kategori == "Sedang" else "-",
                        "DIREKOMENDASIKAN" if kategori == "Sedikit" else "-",
                    ],
                }
            )

            ranking_df = ranking_df.sort_values("Nilai Fuzzy", ascending=False)
            ranking_df.index = range(1, len(ranking_df) + 1)
            ranking_df.index.name = "Peringkat"

            st.dataframe(ranking_df, width="stretch")

            st.subheader("Ringkasan Input")
            ringkasan = pd.DataFrame(
                {
                    "Parameter": [
                        "Suhu Udara",
                        "Kelembaban Tanah",
                        "Curah Hujan (aktual)",
                        "Curah Hujan (norm.)",
                        "Kelembaban Udara",
                        "Jenis Tanaman",
                        "Jenis Tanah",
                        "Tahap Pertumbuhan",
                    ],
                    "Nilai": [
                        f"{temperature} C",
                        f"{soil_moisture}%",
                        f"{rainfall_raw} mm/hari",
                        f"{rainfall_norm}/100",
                        f"{humidity}%",
                        crop_type,
                        soil_type,
                        growth_stage,
                    ],
                }
            )
            st.dataframe(ringkasan, width="stretch", hide_index=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat komputasi: {e}")
            st.info(
                "Coba ubah nilai input, beberapa kombinasi mungkin tidak tercakup oleh rule manapun."
            )


# page visualisasi

elif menu == "Visualisasi":

    st.title("Visualisasi Data Irigasi")

    # grafik 1: bar chart distribusi irrigation need

    st.subheader("1. Distribusi Kebutuhan Irigasi")

    counts = df["Irrigation_Need"].value_counts()

    fig1, ax1 = plt.subplots(figsize=(6, 4))

    bars = ax1.bar(counts.index, counts.values, color=["#2196F3", "#4CAF50", "#FF5722"])

    for bar, val in zip(bars, counts.values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            str(val),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax1.set_title("Distribusi Kelas Kebutuhan Irigasi", fontweight="bold")
    ax1.set_xlabel("Kategori")
    ax1.set_ylabel("Jumlah Data")
    ax1.grid(True, linestyle="-.", alpha=0.3)
    fig1.tight_layout()
    st.pyplot(fig1)

    st.divider()

    # grafik 2: bar chart rata-rata suhu per jenis tanaman

    st.subheader("2. Rata-rata Suhu per Jenis Tanaman")

    avg_temp = (
        df.groupby("Crop_Type")["Temperature_C"].mean().sort_values(ascending=False)
    )

    fig2, ax2 = plt.subplots(figsize=(8, 4))

    ax2.bar(avg_temp.index, avg_temp.values, color="#42eff5")

    for i, (crop, val) in enumerate(avg_temp.items()):
        ax2.text(i, val + 0.2, f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax2.set_title("Rata-rata Suhu Udara per Jenis Tanaman", fontweight="bold")
    ax2.set_xlabel("Jenis Tanaman")
    ax2.set_ylabel("Rata-rata Suhu (C)")
    ax2.grid(True, linestyle="-.", alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2)

    st.divider()

    # grafik 3: scatter plot Kelembaban tanah vs suhu
    # bisa kita lihat di grafik ini,
    # titik merah atau kategori high hampir semuanya berada di bawah kelembaban tanah 25%
    # tersebar di berbagai rentang suhu. ini memvalidasi rule fuzzy kita,
    # khususnya R2 dan R10 yang menyatakan bahwa tanah kering adalah faktor kuat
    # penentu kebutuhan irigasi tinggi, terlepas dari suhunya.
    st.subheader("3. Kelembaban Tanah vs Suhu Udara")

    sample = df.sample(n=min(500, len(df)), random_state=42)

    color_map = {"Low": "#2196F3", "Medium": "#4CAF50", "High": "#FF5722"}

    fig3, ax3 = plt.subplots(figsize=(7, 4))

    for label, color in color_map.items():
        subset = sample[sample["Irrigation_Need"] == label]
        ax3.scatter(
            subset["Temperature_C"],
            subset["Soil_Moisture"],
            color=color,
            label=label,
            alpha=0.6,
            marker="o",
        )

    ax3.set_title("Kelembaban Tanah vs Suhu (sample 500 data)", fontweight="bold")
    ax3.set_xlabel("Suhu (C)")
    ax3.set_ylabel("Kelembaban Tanah (%)")
    ax3.legend(title="Kebutuhan Irigasi")
    ax3.grid(True, linestyle="--", alpha=0.3)
    fig3.tight_layout()
    st.pyplot(fig3)

    st.divider()

    # grafik 4: pie chart proporsi jenis tanah

    st.subheader("4. Proporsi Jenis Tanah dalam Dataset")

    soil_counts = df["Soil_Type"].value_counts()

    fig4, ax4 = plt.subplots(figsize=(6, 6))

    ax4.pie(
        soil_counts.values,
        labels=soil_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#42eff5", "#7542f5", "#f5a742", "#42f56f", "#f54242"],
    )

    ax4.set_title("Persentase Jenis Tanah dalam Dataset", fontweight="bold")
    fig4.tight_layout()
    st.pyplot(fig4)

    st.divider()

    # grafik 5: subplot - scatter curah hujan vs suhu dan bar rata-rata kelembaban per tahap
    # subplot kiri: scatter curah hujan vs suhu
    # scatter plot kiri menunjukkan hubungan antara curah hujan dan suhu dari 500 sampel data.
    # titik-titik tersebar merata tanpa pola kluster yang jelas,
    # artinya tidak ada korelasi kuat antara suhu dan curah hujan dalam dataset ini.

    st.subheader("5. Perbandingan Dua Grafik (Subplot)")

    avg_moisture = df.groupby("Crop_Growth_Stage")["Soil_Moisture"].mean().sort_values()

    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(12, 4))

    ax5a.scatter(
        sample["Rainfall_mm"],
        sample["Temperature_C"],
        color="red",
        alpha=0.4,
        marker="o",
    )
    ax5a.set_title("Curah Hujan vs Suhu")
    ax5a.set_xlabel("Curah Hujan (mm)")
    ax5a.set_ylabel("Suhu (C)")
    ax5a.grid(True, linestyle="--", alpha=0.3)

    # Subplot kanan: bar rata-rata kelembaban per tahap
    ax5b.bar(avg_moisture.index, avg_moisture.values, color="skyblue")
    ax5b.set_title("Rata-rata Kelembaban Tanah per Tahap")
    ax5b.set_xlabel("Tahap Pertumbuhan")
    ax5b.set_ylabel("Rata-rata Kelembaban Tanah (%)")
    ax5b.grid(True, linestyle="-.", alpha=0.3)

    fig5.tight_layout()
    st.pyplot(fig5)


# page profil kelompok

elif menu == "Profil Kelompok":

    st.title("Profil Kelompok")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Anggota Kelompok")
        st.markdown("""
        | No | Nama | NIM |
        |----|------|-----|
        | 1  | Wildan Rifqi Prambudi | 123240060 |
        | 2  | Muhammad Fariel Zaky Faheza | 123240077 |
        """)

        st.subheader("Mata Kuliah")
        st.write("Praktikum Sistem Cerdas Pendukung Keputusan (SCPK)")

        st.subheader("Kelas")
        st.write("IF-A / 2025-2026")

    with col2:
        st.subheader("Detail Proyek")
        st.markdown("""
        | Atribut | Detail |
        |---------|--------|
        | Judul | SPK Irigasi Berbasis Fuzzy |
        | Metode | Fuzzy Mamdani |
        | Dataset | Irrigation Prediction CSV |
        | Jumlah Data | 10.000 baris |
        | Jumlah Kriteria | 7 |
        | Tahun | 2025/2026 |
        """)

        st.subheader("Tools yang Digunakan")
        st.markdown("""
        - Python 3.x
        - Streamlit
        - Scikit-Fuzzy
        - Pandas & NumPy
        - Matplotlib
        """)
