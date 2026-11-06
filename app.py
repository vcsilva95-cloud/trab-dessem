from __future__ import annotations

import io
import json
from pathlib import Path
from dataclasses import asdict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modelo import (
    DT_H,
    N_PERIODOS,
    PATAMARES_DIA,
    aplicar_entradas,
    curva_carga_semanal,
    parametros_padrao,
    resolver,
)

st.set_page_config(
    page_title="Programacao Hidrotermica Semanal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Garantia visual adicional, inclusive para componentes que nao herdam o tema.
# Tema claro forcado para todos os componentes do Streamlit
st.markdown(
    """
    <style>

    /* Pagina e barra lateral */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }

    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"] {
        background-color: #F7F7F7 !important;
        color: #000000 !important;
    }

    /* Textos gerais */
    html, body, p, label, span,
    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] {
        color: #000000 !important;
    }

    /* Campos numericos e de texto */
    input,
    textarea {
        color: #000000 !important;
        background-color: #FFFFFF !important;
        -webkit-text-fill-color: #000000 !important;
    }

    /* Number input completo */
    [data-testid="stNumberInput"] > div,
    [data-testid="stNumberInput"] div[data-baseweb="input"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }

    /* Botoes + e - dos campos numericos */
    [data-testid="stNumberInput"] button {
        background-color: #EEEEEE !important;
        color: #000000 !important;
        border-color: #BDBDBD !important;
    }

    [data-testid="stNumberInput"] button:hover {
        background-color: #DADADA !important;
        color: #000000 !important;
    }

    /* Selectbox */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border-color: #BDBDBD !important;
    }

    div[data-baseweb="select"] span {
        color: #000000 !important;
    }

    /* Botoes comuns */
    .stButton > button {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #666666 !important;
    }

    .stButton > button:hover {
        background-color: #EAEAEA !important;
        color: #000000 !important;
        border-color: #000000 !important;
    }

    /* Botao de download */
    .stDownloadButton > button {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #666666 !important;
    }

    .stDownloadButton > button:hover {
        background-color: #EAEAEA !important;
        color: #000000 !important;
        border-color: #000000 !important;
    }

    /* Expanders das termicas */
    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border-color: #999999 !important;
    }

    [data-testid="stExpander"] summary {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }

    /* Radio buttons */
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] p {
        color: #000000 !important;
    }

    /* Checkbox */
    [data-testid="stCheckbox"] label,
    [data-testid="stCheckbox"] p {
        color: #000000 !important;
    }

    /* Upload de CSV */
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #666666 !important;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }

    /* Alertas do Streamlit */
    /* Caixas de aviso, informacao, sucesso e alerta */
    [data-testid="stAlert"] {
      background-color: #F2F2F2 !important;
      color: #000000 !important;
      border: 1px solid #BDBDBD !important;
    }

    [data-testid="stAlert"] p,
    [data-testid="stAlert"] div,
    [data-testid="stAlert"] span {
      color: #000000 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

PRETO = "#000000"
BRANCO = "#FFFFFF"
CORES = {
    "Carga": "#000000",
    "UHE Chavantes": "#1F77B4",
    "Termobahia UG1": "#D62728",
    "Campina Grande UG1": "#FF7F0E",
    "Deficit": "#9467BD",
    "Volume": "#2CA02C",
}


def estilo_figura(fig: go.Figure, titulo: str, yaxis: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=titulo, font=dict(color=PRETO, size=20)),
        paper_bgcolor=BRANCO,
        plot_bgcolor=BRANCO,
        font=dict(color=PRETO),
        legend=dict(
            font=dict(color=PRETO),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=PRETO,
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        xaxis=dict(
            title="Periodo de 30 minutos",
            color=PRETO,
            tickfont=dict(color=PRETO),
            gridcolor="#D9D9D9",
            linecolor=PRETO,
            zerolinecolor=PRETO,
        ),
        yaxis=dict(
            title=yaxis,
            color=PRETO,
            tickfont=dict(color=PRETO),
            gridcolor="#D9D9D9",
            linecolor=PRETO,
            zerolinecolor=PRETO,
        ),
        hoverlabel=dict(bgcolor=BRANCO, font_color=PRETO),
        margin=dict(l=60, r=30, t=95, b=60),
    )
    return fig


def numero(rotulo, valor, minimo=0.0, passo=0.01, formato="%.2f", ajuda=None):
    return st.number_input(
        rotulo,
        min_value=float(minimo),
        value=float(valor),
        step=float(passo),
        format=formato,
        help=ajuda,
    )


def editor_termica(ute, chave: str) -> dict:
    with st.expander(ute.nome, expanded=False):
        pmin = numero("Potencia minima (MW)", ute.pmin_mw, passo=0.01, key_fix=None) if False else st.number_input(
            "Potencia minima (MW)", min_value=0.0, value=float(ute.pmin_mw), step=0.01, key=f"{chave}_pmin"
        )
        pmax = st.number_input("Potencia maxima (MW)", min_value=0.0, value=float(ute.pmax_mw), step=0.01, key=f"{chave}_pmax")
        ton = st.number_input("Tempo minimo ligada, TON (h)", min_value=0.0, value=float(ute.ton_h), step=0.5, key=f"{chave}_ton")
        toff = st.number_input("Tempo minimo desligada, TOFF (h)", min_value=0.0, value=float(ute.toff_h), step=0.5, key=f"{chave}_toff")
        rup = st.number_input("Rampa de tomada (MW/h)", min_value=0.0, value=float(ute.rampa_subida_mw_h), step=1.0, key=f"{chave}_rup")
        rdn = st.number_input("Rampa de alivio (MW/h)", min_value=0.0, value=float(ute.rampa_descida_mw_h), step=1.0, key=f"{chave}_rdn")
        cvu = st.number_input("CVU (R$/MWh)", min_value=0.0, value=float(ute.cvu_rs_mwh), step=1.0, key=f"{chave}_cvu")
        cpart = st.number_input("Custo de partida (R$)", min_value=0.0, value=float(ute.custo_partida_rs), step=100.0, key=f"{chave}_cpart")
        cpar = st.number_input("Custo de parada (R$)", min_value=0.0, value=float(ute.custo_parada_rs), step=100.0, key=f"{chave}_cpar")
        traj_part = st.text_input(
            "Trajetoria de partida (MW, separada por virgulas)",
            value=", ".join(str(v) for v in ute.trajetoria_partida_mw),
            key=f"{chave}_trajp",
        )
        traj_par = st.text_input(
            "Trajetoria de parada (MW, separada por virgulas)",
            value=", ".join(str(v) for v in ute.trajetoria_parada_mw),
            key=f"{chave}_trajs",
        )
        ligada = st.selectbox("Estado inicial", [0, 1], index=int(ute.ligada_inicial), format_func=lambda x: "Ligada" if x else "Desligada", key=f"{chave}_lig")
        ger_ini = st.number_input("Geracao inicial (MW)", min_value=0.0, value=float(ute.geracao_inicial_mw), step=0.01, key=f"{chave}_gini")
    def lista(txt):
        return [float(x.strip().replace(",", ".")) for x in txt.split(";") if x.strip()] if ";" in txt else [float(x.strip()) for x in txt.split(",") if x.strip()]
    return {
        "nome": ute.nome, "pmin_mw": pmin, "pmax_mw": pmax,
        "ton_h": ton, "toff_h": toff,
        "rampa_subida_mw_h": rup, "rampa_descida_mw_h": rdn,
        "cvu_rs_mwh": cvu, "custo_partida_rs": cpart,
        "custo_parada_rs": cpar,
        "trajetoria_partida_mw": lista(traj_part),
        "trajetoria_parada_mw": lista(traj_par),
        "ligada_inicial": ligada, "geracao_inicial_mw": ger_ini,
    }


base = parametros_padrao()
st.title("Programacao semanal hidrotermica")
st.caption("UHE Chavantes + Termobahia UG1 + Campina Grande UG1 | MILP resolvido com PySCIPOpt")

with st.sidebar:
    st.header("Dados editaveis do sistema")
    st.caption("Horizonte fixo: 168 h, 336 periodos de 30 min e 48 patamares por dia.")

    st.subheader("Carga")
    media_carga = st.number_input("Carga media semanal (MW)", min_value=1.0, value=500.0, step=10.0)
    modo_carga = st.radio("Forma de entrada da carga", ["Perfil sintetico", "Editar tabela", "Enviar CSV"])
    carga = curva_carga_semanal(media_carga)
    if modo_carga == "Editar tabela":
        tabela_carga = pd.DataFrame({"periodo": range(N_PERIODOS), "carga_mw": carga})
        tabela_carga = st.data_editor(tabela_carga, hide_index=True, disabled=["periodo"], height=260)
        carga = tabela_carga["carga_mw"].astype(float).tolist()
    elif modo_carga == "Enviar CSV":
        arquivo = st.file_uploader("CSV com coluna carga_mw e 336 linhas", type=["csv"])
        if arquivo is not None:
            recebido = pd.read_csv(arquivo)
            if "carga_mw" not in recebido.columns or len(recebido) != N_PERIODOS:
                st.error("O CSV deve conter a coluna carga_mw e exatamente 336 linhas.")
            else:
                carga = recebido["carga_mw"].astype(float).tolist()

    st.subheader("UHE Chavantes")
    vi = st.number_input("Volume inicial (hm3)", min_value=float(base.uhe.volume_min_hm3), max_value=float(base.uhe.volume_max_hm3), value=float(base.uhe.volume_inicial_hm3), step=10.0)
    afl = st.number_input("Vazao afluente constante (m3/s)", min_value=0.0, value=float(base.uhe.vazao_afluente_m3s), step=10.0)
    seguir_meta = st.checkbox("Seguir meta de reservatorio", value=False, help="Quando marcada, a meta entra no modelo com penalidade por desvio.")
    meta = st.number_input("Meta de volume final (hm3)", min_value=float(base.uhe.volume_min_hm3), max_value=float(base.uhe.volume_max_hm3), value=float(base.uhe.volume_inicial_hm3), step=10.0)

    st.subheader("Termicas")
    termicas = [
        editor_termica(base.termicas[0], "tb"),
        editor_termica(base.termicas[1], "cg"),
    ]

    st.subheader("Funcao de custo futuro")
    considerar_fcf = st.radio("Considerar FCF?", ["Nao", "Sim"], horizontal=True, help="Calcula uma FCF mensal por PL unico continuo e a inclui no custo do MILP semanal.") == "Sim"
    pontos_fcf = st.slider("Pontos da curva da FCF", min_value=5, max_value=41, value=21, step=2, disabled=not considerar_fcf)
    st.subheader("Custos e solver")
    cdef = st.number_input("Custo de deficit (R$/MWh)", min_value=0.0, value=float(base.custo_deficit_rs_mwh), step=100.0)
    gap = st.number_input("Gap relativo do SCIP", min_value=0.0, max_value=1.0, value=float(base.gap_relativo), step=0.0001, format="%.4f")
    limite = st.number_input("Limite de tempo do SCIP (s, 0 = sem limite)", min_value=0, value=0, step=10)
    executar = st.button("Executar otimizacao", type="primary", use_container_width=True)

entradas = {
    "carga_mw": carga,
    "custo_deficit_rs_mwh": cdef,
    "gap_relativo": gap,
    "limite_tempo_s": None if limite == 0 else float(limite),
    "exibir_log_solver": False,
    "considerar_fcf": considerar_fcf,
    "numero_pontos_fcf": pontos_fcf,
    "uhe": {
        "volume_inicial_hm3": vi,
        "volume_meta_final_hm3": meta,
        "cumprir_meta_reservatorio": seguir_meta,
        "vazao_afluente_m3s": afl,
    },
    "termicas": termicas,
}

st.download_button(
    "Baixar entradas em JSON",
    data=json.dumps(entradas, ensure_ascii=False, indent=2),
    file_name="entradas.json",
    mime="application/json",
)

if executar:
    try:
        sistema = aplicar_entradas(base, entradas)
        with st.spinner("Resolvendo o MILP no SCIP..."):
            st.session_state["resultado"] = resolver(sistema)
    except Exception as erro:
        st.exception(erro)

resultado = st.session_state.get("resultado")
if not resultado:
    st.info("Ajuste os dados na barra lateral e clique em Executar otimizacao.")
    st.stop()

if "resultados" not in resultado:
    st.error(resultado.get("mensagem", "Nao foi encontrada solucao."))
    st.write("Status do SCIP:", resultado.get("status"))
    st.stop()

df = resultado["resultados"].copy()
for coluna in ["deficit_mw", "excesso_mw"]:
    df.loc[df[coluna].abs() < 1e-6, coluna] = 0.0
df["hora_semana"] = df["periodo"] * DT_H

m1, m2, m3, m4 = st.columns(4)
m1.metric("Status SCIP", resultado["status"])
m2.metric("Custo total", f"R$ {resultado['custo_total_rs']:,.2f}")
m3.metric("Gap", f"{100*resultado['gap']:.4f}%")
m4.metric("Volume final", f"{resultado['volume_final_hm3']:.2f} hm3")
if resultado.get("meta_reservatorio_ativa"):
    if resultado.get("meta_reservatorio_atendida"):
        st.success(f"Meta de reservatorio atendida: volume final = {resultado['volume_final_hm3']:.2f} hm3.")
    else:
        st.warning(f"Nao foi possivel atender integralmente a meta de {resultado['meta_reservatorio_hm3']:.2f} hm3. Desvio = {resultado['desvio_meta_hm3']:.2f} hm3.")
if resultado.get("considerar_fcf"):
    st.info(f"FCF considerada no custo total: R$ {resultado['custo_futuro_rs']:,.2f}.")

col_tb = "termobahia_ug1_mw"
col_cg = "campina_grande_ug1_mw"

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df["hora_semana"], y=df["carga_mw"], name="Carga", line=dict(color=CORES["Carga"], width=3)))
fig1.add_trace(go.Scatter(x=df["hora_semana"], y=df["uhe_mw"], name="UHE Chavantes", stackgroup="geracao", line=dict(color=CORES["UHE Chavantes"])))
fig1.add_trace(go.Scatter(x=df["hora_semana"], y=df[col_tb], name="Termobahia UG1", stackgroup="geracao", line=dict(color=CORES["Termobahia UG1"])))
fig1.add_trace(go.Scatter(x=df["hora_semana"], y=df[col_cg], name="Campina Grande UG1", stackgroup="geracao", line=dict(color=CORES["Campina Grande UG1"])))
estilo_figura(fig1, "Despacho de geracao e carga", "Potencia (MW)")
st.plotly_chart(fig1, width="stretch", theme=None)

c1, c2 = st.columns(2)
with c1:
    fig2 = go.Figure(go.Scatter(x=df["hora_semana"], y=df["volume_hm3"], name="Volume", line=dict(color=CORES["Volume"], width=3)))
    estilo_figura(fig2, "Volume do reservatorio", "Volume (hm3)")
    st.plotly_chart(fig2, width="stretch", theme=None)
with c2:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df["hora_semana"], y=df["deficit_mw"], name="Deficit", line=dict(color=CORES["Deficit"], width=3)))
    fig3.add_trace(go.Scatter(x=df["hora_semana"], y=df["excesso_mw"], name="Excesso", line=dict(color="#7F7F7F", width=2)))
    estilo_figura(fig3, "Deficit e excesso", "Potencia (MW)")
    fig3.update_yaxes(tickformat=".3f", rangemode="tozero")
    st.plotly_chart(fig3, width="stretch", theme=None)

st.subheader("Resultados por periodo")
st.dataframe(df, hide_index=True, width="stretch", height=420)

st.download_button(
    "Baixar resultados CSV",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="despacho_30min.csv",
    mime="text/csv",
)